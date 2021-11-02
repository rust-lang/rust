//! Representation of a `TextEdit`.
//!
//! `rust-analyzer` never mutates text itself and only sends diffs to clients,
//! so `TextEdit` is the ultimate representation of the work done by
//! rust-analyzer.
use std::collections::HashSet;

pub use text_size::{TextRange, TextSize};

/// `InsertDelete` -- a single "atomic" change to text
///
/// Must not overlap with other `InDel`s
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Indel {
    pub insert: String,
    /// Refers to offsets in the original text
    pub delete: TextRange,
}

#[derive(Default, Debug, Clone)]
pub struct TextEdit {
    /// Invariant: disjoint and sorted by `delete`.
    indels: Vec<Indel>,
}

#[derive(Debug, Default, Clone)]
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
    pub fn builder() -> TextEditBuilder {
        TextEditBuilder::default()
    }

    pub fn insert(offset: TextSize, text: String) -> TextEdit {
        let mut builder = TextEdit::builder();
        builder.insert(offset, text);
        builder.finish()
    }

    pub fn delete(range: TextRange) -> TextEdit {
        let mut builder = TextEdit::builder();
        builder.delete(range);
        builder.finish()
    }

    pub fn replace(range: TextRange, replace_with: String) -> TextEdit {
        let mut builder = TextEdit::builder();
        builder.replace(range, replace_with);
        builder.finish()
    }

    pub fn len(&self) -> usize {
        self.indels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indels.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Indel> {
        self.into_iter()
    }

    pub fn apply(&self, text: &mut String) {
        match self.len() {
            0 => return,
            1 => {
                self.indels[0].apply(text);
                return;
            }
            _ => (),
        }

        let mut total_len = TextSize::of(&*text);
        for indel in &self.indels {
            total_len += TextSize::of(&indel.insert);
            total_len -= indel.delete.end() - indel.delete.start();
        }
        let mut buf = String::with_capacity(total_len.into());
        let mut prev = 0;
        for indel in &self.indels {
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
        *text = buf;
    }

    pub fn union(&mut self, other: TextEdit) -> Result<(), TextEdit> {
        dbg!(&self, &other);
        // FIXME: can be done without allocating intermediate vector
        let mut all = self.iter().chain(other.iter()).collect::<Vec<_>>();
        if !check_disjoint_or_equal(&mut all) {
            return Err(other);
        }

        // remove duplicates
        // FIXME: maybe make indels a HashSet instead to get rid of this?
        let our_indels = self.indels.clone();
        let our_indels = our_indels.iter().collect::<HashSet<_>>();
        let other_indels = other.indels.into_iter().filter(|i| !our_indels.contains(i));

        self.indels.extend(other_indels);
        Ok(())
    }

    pub fn apply_to_offset(&self, offset: TextSize) -> Option<TextSize> {
        let mut res = offset;
        for indel in &self.indels {
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

impl IntoIterator for TextEdit {
    type Item = Indel;
    type IntoIter = std::vec::IntoIter<Indel>;

    fn into_iter(self) -> Self::IntoIter {
        self.indels.into_iter()
    }
}

impl<'a> IntoIterator for &'a TextEdit {
    type Item = &'a Indel;
    type IntoIter = std::slice::Iter<'a, Indel>;

    fn into_iter(self) -> Self::IntoIter {
        self.indels.iter()
    }
}

impl TextEditBuilder {
    pub fn is_empty(&self) -> bool {
        self.indels.is_empty()
    }
    pub fn replace(&mut self, range: TextRange, replace_with: String) {
        self.indel(Indel::replace(range, replace_with));
    }
    pub fn delete(&mut self, range: TextRange) {
        self.indel(Indel::delete(range));
    }
    pub fn insert(&mut self, offset: TextSize, text: String) {
        self.indel(Indel::insert(offset, text));
    }
    pub fn finish(self) -> TextEdit {
        let mut indels = self.indels;
        assert_disjoint_or_equal(&mut indels);
        TextEdit { indels }
    }
    pub fn invalidates_offset(&self, offset: TextSize) -> bool {
        self.indels.iter().any(|indel| indel.delete.contains_inclusive(offset))
    }
    fn indel(&mut self, indel: Indel) {
        self.indels.push(indel);
        if self.indels.len() <= 16 {
            assert_disjoint_or_equal(&mut self.indels);
        }
    }
}

fn assert_disjoint_or_equal(indels: &mut [impl std::borrow::Borrow<Indel>]) {
    assert!(check_disjoint_or_equal(indels));
}
fn check_disjoint_or_equal(indels: &mut [impl std::borrow::Borrow<Indel>]) -> bool {
    indels.sort_by_key(|indel| (indel.borrow().delete.start(), indel.borrow().delete.end()));
    indels.iter().zip(indels.iter().skip(1)).all(|(l, r)| {
        let l = l.borrow();
        let r = r.borrow();
        l.delete.end() <= r.delete.start() || l == r
    })
}
