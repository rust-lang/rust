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
use text_edit::TextEdit;

#[derive(Default, Debug, Clone)]
pub struct SourceChange {
    pub source_file_edits: SourceFileEdits,
    pub file_system_edits: Vec<FileSystemEdit>,
    pub is_snippet: bool,
}

impl SourceChange {
    /// Creates a new SourceChange with the given label
    /// from the edits.
    pub fn from_edits(
        source_file_edits: SourceFileEdits,
        file_system_edits: Vec<FileSystemEdit>,
    ) -> Self {
        SourceChange { source_file_edits, file_system_edits, is_snippet: false }
    }
}

#[derive(Default, Debug, Clone)]
pub struct SourceFileEdits {
    pub edits: FxHashMap<FileId, TextEdit>,
}

impl SourceFileEdits {
    pub fn from_text_edit(file_id: FileId, edit: TextEdit) -> Self {
        SourceFileEdits { edits: FxHashMap::from_iter(iter::once((file_id, edit))) }
    }

    pub fn len(&self) -> usize {
        self.edits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    pub fn insert(&mut self, file_id: FileId, edit: TextEdit) {
        match self.edits.entry(file_id) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().union(edit).expect("overlapping edits for same file");
            }
            Entry::Vacant(entry) => {
                entry.insert(edit);
            }
        }
    }
}

impl Extend<(FileId, TextEdit)> for SourceFileEdits {
    fn extend<T: IntoIterator<Item = (FileId, TextEdit)>>(&mut self, iter: T) {
        iter.into_iter().for_each(|(file_id, edit)| self.insert(file_id, edit));
    }
}

impl From<SourceFileEdits> for SourceChange {
    fn from(source_file_edits: SourceFileEdits) -> SourceChange {
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
