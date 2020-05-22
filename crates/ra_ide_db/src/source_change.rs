//! This modules defines type to represent changes to the source code, that flow
//! from the server to the client.
//!
//! It can be viewed as a dual for `AnalysisChange`.

use ra_db::{FileId, RelativePathBuf, SourceRootId};
use ra_text_edit::TextEdit;

#[derive(Debug, Clone)]
pub struct SourceChange {
    pub source_file_edits: Vec<SourceFileEdit>,
    pub file_system_edits: Vec<FileSystemEdit>,
    pub is_snippet: bool,
}

impl SourceChange {
    /// Creates a new SourceChange with the given label
    /// from the edits.
    pub fn from_edits(
        source_file_edits: Vec<SourceFileEdit>,
        file_system_edits: Vec<FileSystemEdit>,
    ) -> Self {
        SourceChange { source_file_edits, file_system_edits, is_snippet: false }
    }

    /// Creates a new SourceChange with the given label,
    /// containing only the given `SourceFileEdits`.
    pub fn source_file_edits(edits: Vec<SourceFileEdit>) -> Self {
        SourceChange { source_file_edits: edits, file_system_edits: vec![], is_snippet: false }
    }
    /// Creates a new SourceChange with the given label
    /// from the given `FileId` and `TextEdit`
    pub fn source_file_edit_from(file_id: FileId, edit: TextEdit) -> Self {
        SourceFileEdit { file_id, edit }.into()
    }
}

#[derive(Debug, Clone)]
pub struct SourceFileEdit {
    pub file_id: FileId,
    pub edit: TextEdit,
}

impl From<SourceFileEdit> for SourceChange {
    fn from(edit: SourceFileEdit) -> SourceChange {
        SourceChange {
            source_file_edits: vec![edit],
            file_system_edits: Vec::new(),
            is_snippet: false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum FileSystemEdit {
    CreateFile { source_root: SourceRootId, path: RelativePathBuf },
    MoveFile { src: FileId, dst_source_root: SourceRootId, dst_path: RelativePathBuf },
}

impl From<FileSystemEdit> for SourceChange {
    fn from(edit: FileSystemEdit) -> SourceChange {
        SourceChange {
            source_file_edits: Vec::new(),
            file_system_edits: vec![edit],
            is_snippet: false,
        }
    }
}
