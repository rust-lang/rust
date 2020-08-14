//! This modules defines type to represent changes to the source code, that flow
//! from the server to the client.
//!
//! It can be viewed as a dual for `AnalysisChange`.

use base_db::FileId;
use text_edit::TextEdit;

#[derive(Default, Debug, Clone)]
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
}

#[derive(Debug, Clone)]
pub struct SourceFileEdit {
    pub file_id: FileId,
    pub edit: TextEdit,
}

impl From<SourceFileEdit> for SourceChange {
    fn from(edit: SourceFileEdit) -> SourceChange {
        vec![edit].into()
    }
}

impl From<Vec<SourceFileEdit>> for SourceChange {
    fn from(source_file_edits: Vec<SourceFileEdit>) -> SourceChange {
        SourceChange { source_file_edits, file_system_edits: Vec::new(), is_snippet: false }
    }
}

#[derive(Debug, Clone)]
pub enum FileSystemEdit {
    CreateFile { anchor: FileId, dst: String },
    MoveFile { src: FileId, anchor: FileId, dst: String },
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
