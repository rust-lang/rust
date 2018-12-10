mod edit;
pub mod text_utils;

pub use crate::edit::{Edit, EditBuilder};

use text_unit::{TextRange, TextUnit};

#[derive(Debug, Clone)]
pub struct AtomEdit {
    pub delete: TextRange,
    pub insert: String,
}

impl AtomEdit {
    pub fn replace(range: TextRange, replace_with: String) -> AtomEdit {
        AtomEdit {
            delete: range,
            insert: replace_with,
        }
    }

    pub fn delete(range: TextRange) -> AtomEdit {
        AtomEdit::replace(range, String::new())
    }

    pub fn insert(offset: TextUnit, text: String) -> AtomEdit {
        AtomEdit::replace(TextRange::offset_len(offset, 0.into()), text)
    }
}
