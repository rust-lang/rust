//! FIXME: write short doc here

use crate::AtomTextEdit;

use text_size::{TextRange, TextSize};

#[derive(Debug, Clone)]
pub struct TextEdit {
    atoms: Vec<AtomTextEdit>,
}

#[derive(Debug, Default)]
pub struct TextEditBuilder {
    atoms: Vec<AtomTextEdit>,
}

impl TextEditBuilder {
    pub fn replace(&mut self, range: TextRange, replace_with: String) {
        self.atoms.push(AtomTextEdit::replace(range, replace_with))
    }
    pub fn delete(&mut self, range: TextRange) {
        self.atoms.push(AtomTextEdit::delete(range))
    }
    pub fn insert(&mut self, offset: TextSize, text: String) {
        self.atoms.push(AtomTextEdit::insert(offset, text))
    }
    pub fn finish(self) -> TextEdit {
        TextEdit::from_atoms(self.atoms)
    }
    pub fn invalidates_offset(&self, offset: TextSize) -> bool {
        self.atoms.iter().any(|atom| atom.delete.contains_inclusive(offset))
    }
}

impl TextEdit {
    pub fn insert(offset: TextSize, text: String) -> TextEdit {
        let mut builder = TextEditBuilder::default();
        builder.insert(offset, text);
        builder.finish()
    }

    pub fn delete(range: TextRange) -> TextEdit {
        let mut builder = TextEditBuilder::default();
        builder.delete(range);
        builder.finish()
    }

    pub fn replace(range: TextRange, replace_with: String) -> TextEdit {
        let mut builder = TextEditBuilder::default();
        builder.replace(range, replace_with);
        builder.finish()
    }

    pub(crate) fn from_atoms(mut atoms: Vec<AtomTextEdit>) -> TextEdit {
        atoms.sort_by_key(|a| (a.delete.start(), a.delete.end()));
        for (a1, a2) in atoms.iter().zip(atoms.iter().skip(1)) {
            assert!(a1.delete.end() <= a2.delete.start())
        }
        TextEdit { atoms }
    }

    pub fn as_atoms(&self) -> &[AtomTextEdit] {
        &self.atoms
    }

    pub fn apply(&self, text: &str) -> String {
        let mut total_len = TextSize::of(text);
        for atom in self.atoms.iter() {
            total_len += TextSize::of(&atom.insert);
            total_len -= atom.delete.end() - atom.delete.start();
        }
        let mut buf = String::with_capacity(total_len.into());
        let mut prev = 0;
        for atom in self.atoms.iter() {
            let start: usize = atom.delete.start().into();
            let end: usize = atom.delete.end().into();
            if start > prev {
                buf.push_str(&text[prev..start]);
            }
            buf.push_str(&atom.insert);
            prev = end;
        }
        buf.push_str(&text[prev..text.len()]);
        assert_eq!(TextSize::of(&buf), total_len);
        buf
    }

    pub fn apply_to_offset(&self, offset: TextSize) -> Option<TextSize> {
        let mut res = offset;
        for atom in self.atoms.iter() {
            if atom.delete.start() >= offset {
                break;
            }
            if offset < atom.delete.end() {
                return None;
            }
            res += TextSize::of(&atom.insert);
            res -= atom.delete.len();
        }
        Some(res)
    }
}
