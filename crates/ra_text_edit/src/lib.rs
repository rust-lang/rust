//! FIXME: write short doc here

mod text_edit;

use text_unit::{TextRange, TextUnit};

pub use crate::text_edit::{TextEdit, TextEditBuilder};

/// Must not overlap with other `AtomTextEdit`s
#[derive(Debug, Clone)]
pub struct AtomTextEdit {
    /// Refers to offsets in the original text
    pub delete: TextRange,
    pub insert: String,
}

impl AtomTextEdit {
    pub fn replace(range: TextRange, replace_with: String) -> AtomTextEdit {
        AtomTextEdit { delete: range, insert: replace_with }
    }

    pub fn delete(range: TextRange) -> AtomTextEdit {
        AtomTextEdit::replace(range, String::new())
    }

    pub fn insert(offset: TextUnit, text: String) -> AtomTextEdit {
        AtomTextEdit::replace(TextRange::offset_len(offset, 0.into()), text)
    }

    pub fn apply(&self, mut text: String) -> String {
        let start = self.delete.start().to_usize();
        let end = self.delete.end().to_usize();
        text.replace_range(start..end, &self.insert);
        text
    }
}
