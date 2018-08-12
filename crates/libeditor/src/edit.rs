use {TextRange, TextUnit};

#[derive(Debug)]
pub struct Edit {
    pub atoms: Vec<AtomEdit>,
}

#[derive(Debug)]
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
        let range = self.translate(range);
        self.atoms.push(AtomEdit { delete: range, insert: replacement })
    }

    pub fn delete(&mut self, range: TextRange) {
        self.replace(range, String::new());
    }

    pub fn insert(&mut self, offset: TextUnit, text: String) {
        self.replace(TextRange::offset_len(offset, 0.into()), text)
    }

    pub fn finish(self) -> Edit {
        Edit { atoms: self.atoms }
    }

    fn translate(&self, range: TextRange) -> TextRange {
        let mut range = range;
        for atom in self.atoms.iter() {
            range = atom.apply_to_range(range)
                .expect("conflicting edits");
        }
        range
    }
}

impl Edit {
    pub fn apply(&self, text: &str) -> String {
        let mut text = text.to_owned();
        for atom in self.atoms.iter() {
            text = atom.apply(&text);
        }
        text
    }
}

impl AtomEdit {
    fn apply(&self, text: &str) -> String {
        let prefix = &text[
            TextRange::from_to(0.into(), self.delete.start())
        ];
        let suffix = &text[
            TextRange::from_to(self.delete.end(), TextUnit::of_str(text))
        ];
        let mut res = String::with_capacity(prefix.len() + self.insert.len() + suffix.len());
        res.push_str(prefix);
        res.push_str(&self.insert);
        res.push_str(suffix);
        res
    }

    fn apply_to_position(&self, pos: TextUnit) -> Option<TextUnit> {
        if pos <= self.delete.start() {
            return Some(pos);
        }
        if pos < self.delete.end() {
            return None;
        }
        Some(pos - self.delete.len() + TextUnit::of_str(&self.insert))
    }

    fn apply_to_range(&self, range: TextRange) -> Option<TextRange> {
        Some(TextRange::from_to(
            self.apply_to_position(range.start())?,
            self.apply_to_position(range.end())?,
        ))
    }
}

