//! This modules defines type to represent changes to the source code, that flow
//! from the server to the client.
//!
//! It can be viewed as a dual for `AnalysisChange`.

use std::{
    collections::hash_map::Entry,
    iter::{self, FromIterator},
};

use base_db::{AnchoredPathBuf, FileId};
use rustc_hash::FxHashMap;
use stdx::never;
use text_edit::TextEdit;

#[derive(Default, Debug, Clone)]
pub struct SourceChange {
    pub source_file_edits: FxHashMap<FileId, TextEdit>,
    pub file_system_edits: Vec<FileSystemEdit>,
    pub is_snippet: bool,
}

impl SourceChange {
    /// Creates a new SourceChange with the given label
    /// from the edits.
    pub fn from_edits(
        source_file_edits: FxHashMap<FileId, TextEdit>,
        file_system_edits: Vec<FileSystemEdit>,
    ) -> Self {
        SourceChange { source_file_edits, file_system_edits, is_snippet: false }
    }

    pub fn from_text_edit(file_id: FileId, edit: TextEdit) -> Self {
        SourceChange {
            source_file_edits: FxHashMap::from_iter(iter::once((file_id, edit))),
            ..Default::default()
        }
    }

    pub fn insert_source_edit(&mut self, file_id: FileId, edit: TextEdit) {
        match self.source_file_edits.entry(file_id) {
            Entry::Occupied(mut entry) => {
                never!(entry.get_mut().union(edit).is_err(), "overlapping edits for same file");
            }
            Entry::Vacant(entry) => {
                entry.insert(edit);
            }
        }
    }

    pub fn push_file_system_edit(&mut self, edit: FileSystemEdit) {
        self.file_system_edits.push(edit);
    }

    pub fn get_source_edit(&self, file_id: FileId) -> Option<&TextEdit> {
        self.source_file_edits.get(&file_id)
    }
}

impl Extend<(FileId, TextEdit)> for SourceChange {
    fn extend<T: IntoIterator<Item = (FileId, TextEdit)>>(&mut self, iter: T) {
        iter.into_iter().for_each(|(file_id, edit)| self.insert_source_edit(file_id, edit));
    }
}

impl From<FxHashMap<FileId, TextEdit>> for SourceChange {
    fn from(source_file_edits: FxHashMap<FileId, TextEdit>) -> SourceChange {
        SourceChange { source_file_edits, file_system_edits: Vec::new(), is_snippet: false }
    }
}

#[derive(Debug, Clone)]
pub enum FileSystemEdit {
    CreateFile { dst: AnchoredPathBuf, initial_contents: String },
    MoveFile { src: FileId, dst: AnchoredPathBuf },
}

impl From<FileSystemEdit> for SourceChange {
    fn from(edit: FileSystemEdit) -> SourceChange {
        SourceChange {
            source_file_edits: Default::default(),
            file_system_edits: vec![edit],
            is_snippet: false,
        }
    }
}
