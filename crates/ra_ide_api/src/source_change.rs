//! This modules defines type to represent changes to the source code, that flow
//! from the server to the client.
//!
//! It can be viewed as a dual for `AnalysisChange`.

use ra_db::RelativePathBuf;
use ra_text_edit::TextEdit;

use crate::{FileId, FilePosition, SourceRootId, TextUnit};

#[derive(Debug)]
pub struct SourceChange {
    pub label: String,
    pub source_file_edits: Vec<SourceFileEdit>,
    pub file_system_edits: Vec<FileSystemEdit>,
    pub cursor_position: Option<FilePosition>,
}

impl SourceChange {
    /// Creates a new SourceChange with the given label
    /// from the edits.
    pub(crate) fn from_edits<L: Into<String>>(
        label: L,
        source_file_edits: Vec<SourceFileEdit>,
        file_system_edits: Vec<FileSystemEdit>,
    ) -> Self {
        SourceChange {
            label: label.into(),
            source_file_edits,
            file_system_edits,
            cursor_position: None,
        }
    }

    /// Creates a new SourceChange with the given label,
    /// containing only the given `SourceFileEdits`.
    pub(crate) fn source_file_edits<L: Into<String>>(label: L, edits: Vec<SourceFileEdit>) -> Self {
        SourceChange {
            label: label.into(),
            source_file_edits: edits,
            file_system_edits: vec![],
            cursor_position: None,
        }
    }

    /// Creates a new SourceChange with the given label,
    /// containing only the given `FileSystemEdits`.
    pub(crate) fn file_system_edits<L: Into<String>>(label: L, edits: Vec<FileSystemEdit>) -> Self {
        SourceChange {
            label: label.into(),
            source_file_edits: vec![],
            file_system_edits: edits,
            cursor_position: None,
        }
    }

    /// Creates a new SourceChange with the given label,
    /// containing only a single `SourceFileEdit`.
    pub(crate) fn source_file_edit<L: Into<String>>(label: L, edit: SourceFileEdit) -> Self {
        SourceChange::source_file_edits(label, vec![edit])
    }

    /// Creates a new SourceChange with the given label
    /// from the given `FileId` and `TextEdit`
    pub(crate) fn source_file_edit_from<L: Into<String>>(
        label: L,
        file_id: FileId,
        edit: TextEdit,
    ) -> Self {
        SourceChange::source_file_edit(label, SourceFileEdit { file_id, edit })
    }

    /// Creates a new SourceChange with the given label
    /// from the given `FileId` and `TextEdit`
    pub(crate) fn file_system_edit<L: Into<String>>(label: L, edit: FileSystemEdit) -> Self {
        SourceChange::file_system_edits(label, vec![edit])
    }

    /// Sets the cursor position to the given `FilePosition`
    pub(crate) fn with_cursor(mut self, cursor_position: FilePosition) -> Self {
        self.cursor_position = Some(cursor_position);
        self
    }

    /// Sets the cursor position to the given `FilePosition`
    pub(crate) fn with_cursor_opt(mut self, cursor_position: Option<FilePosition>) -> Self {
        self.cursor_position = cursor_position;
        self
    }
}

#[derive(Debug)]
pub struct SourceFileEdit {
    pub file_id: FileId,
    pub edit: TextEdit,
}

#[derive(Debug)]
pub enum FileSystemEdit {
    CreateFile { source_root: SourceRootId, path: RelativePathBuf },
    MoveFile { src: FileId, dst_source_root: SourceRootId, dst_path: RelativePathBuf },
}

pub(crate) struct SingleFileChange {
    pub label: String,
    pub edit: TextEdit,
    pub cursor_position: Option<TextUnit>,
}

impl SingleFileChange {
    pub(crate) fn into_source_change(self, file_id: FileId) -> SourceChange {
        SourceChange {
            label: self.label,
            source_file_edits: vec![SourceFileEdit { file_id, edit: self.edit }],
            file_system_edits: Vec::new(),
            cursor_position: self.cursor_position.map(|offset| FilePosition { file_id, offset }),
        }
    }
}
