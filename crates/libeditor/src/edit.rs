use {TextRange, TextUnit};
use libsyntax2::AtomEdit;

#[derive(Debug, Clone)]
pub struct Edit {
    atoms: Vec<AtomEdit>,
}

#[derive(Debug)]
pub struct EditBuilder {
    atoms: Vec<AtomEdit>
}

impl EditBuilder {
    pub fn new() -> EditBuilder {
        EditBuilder { atoms: Vec::new() }
    }

    pub fn replace(&mut self, range: TextRange, replace_with: String) {
        self.atoms.push(AtomEdit::replace(range, replace_with))
    }

    pub fn delete(&mut self, range: TextRange) {
        self.atoms.push(AtomEdit::delete(range))
    }

    pub fn insert(&mut self, offset: TextUnit, text: String) {
        self.atoms.push(AtomEdit::insert(offset, text))
    }

    pub fn finish(self) -> Edit {
        let mut atoms = self.atoms;
        atoms.sort_by_key(|a| a.delete.start());
        for (a1, a2) in atoms.iter().zip(atoms.iter().skip(1)) {
            assert!(a1.delete.end() <= a2.delete.start())
        }
        Edit { atoms }
    }
}

impl Edit {
    pub fn into_atoms(self) -> Vec<AtomEdit> {
        self.atoms
    }

    pub fn apply(&self, text: &str) -> String {
        let mut total_len = text.len();
        for atom in self.atoms.iter() {
            total_len += atom.insert.len();
            total_len -= u32::from(atom.delete.end() - atom.delete.start()) as usize;
        }
        let mut buf = String::with_capacity(total_len);
        let mut prev = 0;
        for atom in self.atoms.iter() {
            let start = u32::from(atom.delete.start()) as usize;
            let end = u32::from(atom.delete.end()) as usize;
            if start > prev {
                buf.push_str(&text[prev..start]);
            }
            buf.push_str(&atom.insert);
            prev = end;
        }
        buf.push_str(&text[prev..text.len()]);
        assert_eq!(buf.len(), total_len);
        buf
    }

    pub fn apply_to_offset(&self, offset: TextUnit) -> Option<TextUnit> {
        let mut res = offset;
        for atom in self.atoms.iter() {
            if atom.delete.start() >= offset {
                break;
            }
            if offset < atom.delete.end() {
                return None
            }
            res += TextUnit::of_str(&atom.insert);
            res -= atom.delete.len();
        }
        Some(res)
    }
}
