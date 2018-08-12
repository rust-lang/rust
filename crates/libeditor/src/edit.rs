use {TextRange, TextUnit};

#[derive(Debug, Clone)]
pub struct Edit {
    atoms: Vec<AtomEdit>,
}

#[derive(Debug, Clone)]
pub struct AtomEdit {
    pub delete: TextRange,
    pub insert: String,
}

#[derive(Debug)]
pub struct EditBuilder {
    atoms: Vec<AtomEdit>
}

impl EditBuilder {
    pub fn new() -> EditBuilder {
        EditBuilder { atoms: Vec::new() }
    }

    pub fn replace(&mut self, range: TextRange, replacement: String) {
        self.atoms.push(AtomEdit { delete: range, insert: replacement })
    }

    pub fn delete(&mut self, range: TextRange) {
        self.replace(range, String::new());
    }

    pub fn insert(&mut self, offset: TextUnit, text: String) {
        self.replace(TextRange::offset_len(offset, 0.into()), text)
    }

    pub fn finish(self) -> Edit {
        let mut atoms = self.atoms;
        atoms.sort_by_key(|a| a.delete.start());
        for (a1, a2) in atoms.iter().zip(atoms.iter().skip(1)) {
            assert!(a1.end() <= a2.start())
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
            total_len -= atom.end() - atom.start();
        }
        let mut buf = String::with_capacity(total_len);
        let mut prev = 0;
        for atom in self.atoms.iter() {
            if atom.start() > prev {
                buf.push_str(&text[prev..atom.start()]);
            }
            buf.push_str(&atom.insert);
            prev = atom.end();
        }
        buf.push_str(&text[prev..text.len()]);
        assert_eq!(buf.len(), total_len);
        buf
    }
}

impl AtomEdit {
    fn start(&self) -> usize {
        u32::from(self.delete.start()) as usize
    }

    fn end(&self) -> usize {
        u32::from(self.delete.end()) as usize
    }
}
