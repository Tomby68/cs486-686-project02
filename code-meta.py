"""Code splitter with enhanced metadata for file paths and line numbers."""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.node_parser.interface import TextSplitter
from llama_index.core.node_parser.node_utils import default_id_func
from llama_index.core.schema import Document, NodeRelationship, TextNode

DEFAULT_CHUNK_LINES = 40
DEFAULT_LINES_OVERLAP = 15
DEFAULT_MAX_CHARS = 1500


class CodeSplitter(TextSplitter):
    """Split code using a AST parser with enhanced metadata for file paths and line numbers.

    Thank you to Kevin Lu / SweepAI for suggesting this elegant code splitting solution.
    https://docs.sweep.dev/blogs/chunking-2m-files
    """

    language: str = Field(
        description="The programming language of the code being split."
    )
    chunk_lines: int = Field(
        default=DEFAULT_CHUNK_LINES,
        description="The number of lines to include in each chunk.",
        gt=0,
    )
    chunk_lines_overlap: int = Field(
        default=DEFAULT_LINES_OVERLAP,
        description="How many lines of code each chunk overlaps with.",
        gt=0,
    )
    max_chars: int = Field(
        default=DEFAULT_MAX_CHARS,
        description="Maximum number of characters per chunk.",
        gt=0,
    )
    filepath: Optional[str] = Field(
        default=None,
        description="Full path to the source file.",
    )
    base_dir: Optional[str] = Field(
        default=None,
        description="Base directory for calculating relative paths.",
    )
    _parser: Any = PrivateAttr()
    _line_offsets: Dict[str, List[int]] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        language: str,
        chunk_lines: int = DEFAULT_CHUNK_LINES,
        chunk_lines_overlap: int = DEFAULT_LINES_OVERLAP,
        max_chars: int = DEFAULT_MAX_CHARS,
        filepath: Optional[str] = None,
        base_dir: Optional[str] = None,
        parser: Any = None,
        callback_manager: Optional[CallbackManager] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> None:
        """Initialize a CodeSplitter with enhanced metadata capabilities."""
        from tree_sitter import Parser  # pants: no-infer-dep

        callback_manager = callback_manager or CallbackManager([])
        id_func = id_func or default_id_func

        super().__init__(
            language=language,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars,
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            id_func=id_func,
            filepath=filepath,
            base_dir=base_dir,
        )

        if parser is None:
            try:
                import tree_sitter_languages  # pants: no-infer-dep

                parser = tree_sitter_languages.get_parser(language)
            except ImportError:
                raise ImportError(
                    "Please install tree_sitter_languages to use CodeSplitter."
                    "Or pass in a parser object."
                )
            except Exception:
                print(
                    f"Could not get parser for language {language}. Check "
                    "https://github.com/grantjenks/py-tree-sitter-languages#license "
                    "for a list of valid languages."
                )
                raise
        if not isinstance(parser, Parser):
            raise ValueError("Parser must be a tree-sitter Parser object.")

        self._parser = parser
        self._line_offsets = {}

    @classmethod
    def from_defaults(
        cls,
        language: str,
        chunk_lines: int = DEFAULT_CHUNK_LINES,
        chunk_lines_overlap: int = DEFAULT_LINES_OVERLAP,
        max_chars: int = DEFAULT_MAX_CHARS,
        filepath: Optional[str] = None,
        base_dir: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        parser: Any = None,
    ) -> "CodeSplitter":
        """Create a CodeSplitter with default values."""
        return cls(
            language=language,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars,
            filepath=filepath,
            base_dir=base_dir,
            callback_manager=callback_manager,
            parser=parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "CodeSplitter"

    def _extract_filename(self, filepath: Optional[str]) -> str:
        """Extract base filename from filepath."""
        if not filepath:
            return "unknown"
        return os.path.basename(filepath)

    def _calculate_relpath(self, filepath: Optional[str], base_dir: Optional[str]) -> str:
        """Calculate relative path based on base_dir."""
        if not filepath:
            return "unknown"
        if not base_dir:
            return filepath
        try:
            return os.path.relpath(filepath, base_dir)
        except ValueError:
            # Handle paths on different drives in Windows
            return filepath

    def _calculate_line_offsets(self, text: str) -> List[int]:
        """Calculate byte offsets for the start of each line."""
        offsets = [0]  # First line starts at offset 0
        for i, char in enumerate(text):
            if char == '\n':
                offsets.append(i + 1)  # Next line starts after newline
        return offsets

    def _byte_to_line(self, byte_offset: int, line_offsets: List[int]) -> int:
        """Convert byte offset to line number (0-based)."""
        for i, offset in enumerate(line_offsets):
            if i + 1 < len(line_offsets) and byte_offset < line_offsets[i + 1]:
                return i
            if i + 1 == len(line_offsets):
                return i
        return len(line_offsets) - 1

    def _chunk_node(
        self, 
        node: Any, 
        text: str, 
        line_offsets: Optional[List[int]] = None,
        last_end: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Enhanced chunking method that tracks line numbers and maintains metadata.
        
        Args:
            node: The tree-sitter node to chunk
            text: The original source code text
            line_offsets: List of byte offsets for each line
            last_end: Last ending byte position
            metadata: Metadata to include with chunks
            
        Returns:
            List of tuples (chunk_text, chunk_metadata)
        """
        metadata = metadata or {}
        line_offsets = line_offsets or self._calculate_line_offsets(text)
        
        new_chunks = []
        current_chunk = ""
        current_chunk_start = last_end
        
        for child in node.children:
            if child.end_byte - child.start_byte > self.max_chars:
                # Child is too big, recursively chunk the child
                if len(current_chunk) > 0:
                    # Create metadata for the current chunk
                    chunk_start_line = self._byte_to_line(current_chunk_start, line_offsets)
                    chunk_end_line = self._byte_to_line(last_end, line_offsets)
                    
                    chunk_metadata = {
                        **metadata,
                        "start_line": chunk_start_line + 1,  # Convert to 1-based line numbering
                        "end_line": chunk_end_line + 1
                    }
                    
                    new_chunks.append((current_chunk, chunk_metadata))
                    
                current_chunk = ""
                current_chunk_start = last_end
                
                # Recursively chunk the child with the same metadata base
                child_chunks = self._chunk_node(
                    child, 
                    text, 
                    line_offsets,
                    last_end,
                    metadata
                )
                new_chunks.extend(child_chunks)
                
            elif (
                len(current_chunk) + child.end_byte - child.start_byte > self.max_chars
            ):
                # Child would make the current chunk too big, so start a new chunk
                if len(current_chunk) > 0:
                    # Create metadata for the current chunk
                    chunk_start_line = self._byte_to_line(current_chunk_start, line_offsets)
                    chunk_end_line = self._byte_to_line(last_end, line_offsets)
                    
                    chunk_metadata = {
                        **metadata,
                        "start_line": chunk_start_line + 1,
                        "end_line": chunk_end_line + 1
                    }
                    
                    new_chunks.append((current_chunk, chunk_metadata))
                
                current_chunk = text[last_end : child.end_byte]
                current_chunk_start = last_end
                
            else:
                current_chunk += text[last_end : child.end_byte]
            
            last_end = child.end_byte
        
        # Handle any remaining chunk
        if len(current_chunk) > 0:
            chunk_start_line = self._byte_to_line(current_chunk_start, line_offsets)
            chunk_end_line = self._byte_to_line(last_end, line_offsets)
            
            chunk_metadata = {
                **metadata,
                "start_line": chunk_start_line + 1,
                "end_line": chunk_end_line + 1
            }
            
            new_chunks.append((current_chunk, chunk_metadata))
        
        return new_chunks

    def _process_text_with_metadata(
        self, 
        text: str, 
        filepath: Optional[str] = None,
        base_dir: Optional[str] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Process text and generate chunks with metadata."""
        # Calculate file path information
        filepath = filepath or self.filepath
        base_dir = base_dir or self.base_dir
        filename = self._extract_filename(filepath)
        relpath = self._calculate_relpath(filepath, base_dir)
        
        # Calculate line offsets
        line_offsets = self._calculate_line_offsets(text)
        
        # Base metadata for all chunks
        base_metadata = {
            "filename": filename,
            "filepath": filepath or "unknown",
            "relpath": relpath
        }
        
        # Parse the code
        tree = self._parser.parse(bytes(text, "utf-8"))
        
        if (
            not tree.root_node.children
            or tree.root_node.children[0].type != "ERROR"
        ):
            # Process chunks with metadata
            chunks_with_metadata = self._chunk_node(
                tree.root_node, 
                text,
                line_offsets,
                0,
                base_metadata
            )
            
            # Strip whitespace from chunk text
            chunks_with_metadata = [
                (chunk.strip(), metadata) 
                for chunk, metadata in chunks_with_metadata
            ]
            
            return chunks_with_metadata
        else:
            raise ValueError(f"Could not parse code with language {self.language}.")

    def split_text(self, text: str) -> List[str]:
        """Split incoming code and return chunks using the AST.
        
        Note: This method is maintained for backward compatibility.
        It does not include the enhanced metadata.
        """
        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ) as event:
            chunks_with_metadata = self._process_text_with_metadata(text)
            chunks = [chunk for chunk, _ in chunks_with_metadata]
            
            event.on_end(
                payload={EventPayload.CHUNKS: chunks},
            )
            
            return chunks

    def get_nodes_from_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[TextNode]:
        """Get nodes from documents with enhanced metadata."""
        from llama_index.core.schema import RelatedNodeInfo
        
        nodes = []
        
        for i, doc in enumerate(documents):
            # Extract file information from document metadata or class defaults
            filepath = doc.metadata.get("filepath", self.filepath)
            base_dir = doc.metadata.get("base_dir", self.base_dir)
            
            # Process text with metadata
            chunks_with_metadata = self._process_text_with_metadata(
                doc.text, filepath, base_dir
            )
            
            # Create nodes with the enhanced metadata
            for j, (chunk_text, chunk_metadata) in enumerate(chunks_with_metadata):
                # Create node with enhanced metadata
                node = TextNode(
                    text=chunk_text,
                    metadata={
                        **doc.metadata,  # Original document metadata
                        **chunk_metadata  # Enhanced metadata with file and line info
                    },
                    id_=self.id_func(j, doc),
                )
                nodes.append(node)
        
        # Handle node relationships using RelatedNodeInfo objects
        if self.include_prev_next_rel:
            for i, node in enumerate(nodes):
                if i > 0:
                    # Create a RelatedNodeInfo object for the previous node
                    node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                        node_id=nodes[i-1].id_,
                        metadata={}
                    )
                if i < len(nodes) - 1:
                    # Create a RelatedNodeInfo object for the next node
                    node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                        node_id=nodes[i+1].id_,
                        metadata={}
                    )
        
        return nodes