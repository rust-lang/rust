mod text_edit;
pub mod text_utils;
pub mod test_utils;

pub use crate::text_edit::{TextEdit, TextEditBuilder};

use text_unit::{TextRange, TextUnit};

/// Must not overlap with other `AtomTextEdit`s
#[derive(Debug, Clone)]
pub struct AtomTextEdit {
    /// Refers to offsets in the original text
    pub delete: TextRange,
    pub insert: String,
}

impl AtomTextEdit {
    pub fn replace(range: TextRange, replace_with: String) -> AtomTextEdit {
        AtomTextEdit {
            delete: range,
            insert: replace_with,
        }
    }

    pub fn delete(range: TextRange) -> AtomTextEdit {
        AtomTextEdit::replace(range, String::new())
    }

    pub fn insert(offset: TextUnit, text: String) -> AtomTextEdit {
        AtomTextEdit::replace(TextRange::offset_len(offset, 0.into()), text)
    }
}
