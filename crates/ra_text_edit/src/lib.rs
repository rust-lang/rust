//! Representation of a `TextEdit`.
//!
//! `rust-analyzer` never mutates text itself and only sends diffs to clients,
//! so `TextEdit` is the ultimate representation of the work done by
//! rust-analyzer.

pub use text_size::{TextRange, TextSize};

/// `InsertDelete` -- a single "atomic" change to text
///
/// Must not overlap with other `InDel`s
#[derive(Debug, Clone)]
pub struct Indel {
    pub insert: String,
    /// Refers to offsets in the original text
    pub delete: TextRange,
}

#[derive(Debug, Clone)]
pub struct TextEdit {
    indels: Vec<Indel>,
}

#[derive(Debug, Default)]
pub struct TextEditBuilder {
    indels: Vec<Indel>,
}

impl Indel {
    pub fn insert(offset: TextSize, text: String) -> Indel {
        Indel::replace(TextRange::empty(offset), text)
    }
    pub fn delete(range: TextRange) -> Indel {
        Indel::replace(range, String::new())
    }
    pub fn replace(range: TextRange, replace_with: String) -> Indel {
        Indel { delete: range, insert: replace_with }
    }

    pub fn apply(&self, text: &mut String) {
        let start: usize = self.delete.start().into();
        let end: usize = self.delete.end().into();
        text.replace_range(start..end, &self.insert);
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

    pub(crate) fn from_indels(mut indels: Vec<Indel>) -> TextEdit {
        indels.sort_by_key(|a| (a.delete.start(), a.delete.end()));
        for (a1, a2) in indels.iter().zip(indels.iter().skip(1)) {
            assert!(a1.delete.end() <= a2.delete.start())
        }
        TextEdit { indels }
    }

    pub fn is_empty(&self) -> bool {
        self.indels.is_empty()
    }

    pub fn as_indels(&self) -> &[Indel] {
        &self.indels
    }

    pub fn apply(&self, text: &mut String) {
        match self.indels.len() {
            0 => return,
            1 => {
                self.indels[0].apply(text);
                return;
            }
            _ => (),
        }

        let mut total_len = TextSize::of(&*text);
        for indel in self.indels.iter() {
            total_len += TextSize::of(&indel.insert);
            total_len -= indel.delete.end() - indel.delete.start();
        }
        let mut buf = String::with_capacity(total_len.into());
        let mut prev = 0;
        for indel in self.indels.iter() {
            let start: usize = indel.delete.start().into();
            let end: usize = indel.delete.end().into();
            if start > prev {
                buf.push_str(&text[prev..start]);
            }
            buf.push_str(&indel.insert);
            prev = end;
        }
        buf.push_str(&text[prev..text.len()]);
        assert_eq!(TextSize::of(&buf), total_len);

        // FIXME: figure out a way to mutate the text in-place or reuse the
        // memory in some other way
        *text = buf
    }

    pub fn apply_to_offset(&self, offset: TextSize) -> Option<TextSize> {
        let mut res = offset;
        for indel in self.indels.iter() {
            if indel.delete.start() >= offset {
                break;
            }
            if offset < indel.delete.end() {
                return None;
            }
            res += TextSize::of(&indel.insert);
            res -= indel.delete.len();
        }
        Some(res)
    }
}

impl TextEditBuilder {
    pub fn replace(&mut self, range: TextRange, replace_with: String) {
        self.indels.push(Indel::replace(range, replace_with))
    }
    pub fn delete(&mut self, range: TextRange) {
        self.indels.push(Indel::delete(range))
    }
    pub fn insert(&mut self, offset: TextSize, text: String) {
        self.indels.push(Indel::insert(offset, text))
    }
    pub fn finish(self) -> TextEdit {
        TextEdit::from_indels(self.indels)
    }
    pub fn invalidates_offset(&self, offset: TextSize) -> bool {
        self.indels.iter().any(|indel| indel.delete.contains_inclusive(offset))
    }
}
