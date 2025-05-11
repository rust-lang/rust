//! Representation of a `TextEdit`.
//!
//! `rust-analyzer` never mutates text itself and only sends diffs to clients,
//! so `TextEdit` is the ultimate representation of the work done by
//! rust-analyzer.

use itertools::Itertools;
pub use span::{TextRange, TextSize};
use std::cmp::max;

use crate::source_change::ChangeAnnotationId;

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
    annotation: Option<ChangeAnnotationId>,
}

#[derive(Debug, Default, Clone)]
pub struct TextEditBuilder {
    indels: Vec<Indel>,
    annotation: Option<ChangeAnnotationId>,
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

        let text_size = TextSize::of(&*text);
        let mut total_len = text_size;
        let mut max_total_len = text_size;
        for indel in &self.indels {
            total_len += TextSize::of(&indel.insert);
            total_len -= indel.delete.len();
            max_total_len = max(max_total_len, total_len);
        }

        if let Some(additional) = max_total_len.checked_sub(text_size) {
            text.reserve(additional.into());
        }

        for indel in self.indels.iter().rev() {
            indel.apply(text);
        }

        assert_eq!(TextSize::of(&*text), total_len);
    }

    pub fn union(&mut self, other: TextEdit) -> Result<(), TextEdit> {
        let iter_merge =
            self.iter().merge_by(other.iter(), |l, r| l.delete.start() <= r.delete.start());
        if !check_disjoint(&mut iter_merge.clone()) {
            return Err(other);
        }

        // Only dedup deletions and replacements, keep all insertions
        self.indels = iter_merge.dedup_by(|a, b| a == b && !a.delete.is_empty()).cloned().collect();
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

    pub(crate) fn set_annotation(&mut self, conflict_annotation: Option<ChangeAnnotationId>) {
        self.annotation = conflict_annotation;
    }

    pub fn change_annotation(&self) -> Option<ChangeAnnotationId> {
        self.annotation
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
        let TextEditBuilder { mut indels, annotation } = self;
        assert_disjoint_or_equal(&mut indels);
        indels = coalesce_indels(indels);
        TextEdit { indels, annotation }
    }
    pub fn invalidates_offset(&self, offset: TextSize) -> bool {
        self.indels.iter().any(|indel| indel.delete.contains_inclusive(offset))
    }
    pub fn indel(&mut self, indel: Indel) {
        self.indels.push(indel);
        if self.indels.len() <= 16 {
            assert_disjoint_or_equal(&mut self.indels);
        }
    }
}

fn assert_disjoint_or_equal(indels: &mut [Indel]) {
    assert!(check_disjoint_and_sort(indels));
}

fn check_disjoint_and_sort(indels: &mut [Indel]) -> bool {
    indels.sort_by_key(|indel| (indel.delete.start(), indel.delete.end()));
    check_disjoint(&mut indels.iter())
}

fn check_disjoint<'a, I>(indels: &mut I) -> bool
where
    I: std::iter::Iterator<Item = &'a Indel> + Clone,
{
    indels.clone().zip(indels.skip(1)).all(|(l, r)| l.delete.end() <= r.delete.start() || l == r)
}

fn coalesce_indels(indels: Vec<Indel>) -> Vec<Indel> {
    indels
        .into_iter()
        .coalesce(|mut a, b| {
            if a.delete.end() == b.delete.start() {
                a.insert.push_str(&b.insert);
                a.delete = TextRange::new(a.delete.start(), b.delete.end());
                Ok(a)
            } else {
                Err((a, b))
            }
        })
        .collect_vec()
}

#[cfg(test)]
mod tests {
    use super::{TextEdit, TextEditBuilder, TextRange};

    fn range(start: u32, end: u32) -> TextRange {
        TextRange::new(start.into(), end.into())
    }

    #[test]
    fn test_apply() {
        let mut text = "_11h1_2222_xx3333_4444_6666".to_owned();
        let mut builder = TextEditBuilder::default();
        builder.replace(range(3, 4), "1".to_owned());
        builder.delete(range(11, 13));
        builder.insert(22.into(), "_5555".to_owned());

        let text_edit = builder.finish();
        text_edit.apply(&mut text);

        assert_eq!(text, "_1111_2222_3333_4444_5555_6666")
    }

    #[test]
    fn test_union() {
        let mut edit1 = TextEdit::delete(range(7, 11));
        let mut builder = TextEditBuilder::default();
        builder.delete(range(1, 5));
        builder.delete(range(13, 17));

        let edit2 = builder.finish();
        assert!(edit1.union(edit2).is_ok());
        assert_eq!(edit1.indels.len(), 3);
    }

    #[test]
    fn test_union_with_duplicates() {
        let mut builder1 = TextEditBuilder::default();
        builder1.delete(range(7, 11));
        builder1.delete(range(13, 17));

        let mut builder2 = TextEditBuilder::default();
        builder2.delete(range(1, 5));
        builder2.delete(range(13, 17));

        let mut edit1 = builder1.finish();
        let edit2 = builder2.finish();
        assert!(edit1.union(edit2).is_ok());
        assert_eq!(edit1.indels.len(), 3);
    }

    #[test]
    fn test_union_panics() {
        let mut edit1 = TextEdit::delete(range(7, 11));
        let edit2 = TextEdit::delete(range(9, 13));
        assert!(edit1.union(edit2).is_err());
    }

    #[test]
    fn test_coalesce_disjoint() {
        let mut builder = TextEditBuilder::default();
        builder.replace(range(1, 3), "aa".into());
        builder.replace(range(5, 7), "bb".into());
        let edit = builder.finish();

        assert_eq!(edit.indels.len(), 2);
    }

    #[test]
    fn test_coalesce_adjacent() {
        let mut builder = TextEditBuilder::default();
        builder.replace(range(1, 3), "aa".into());
        builder.replace(range(3, 5), "bb".into());

        let edit = builder.finish();
        assert_eq!(edit.indels.len(), 1);
        assert_eq!(edit.indels[0].insert, "aabb");
        assert_eq!(edit.indels[0].delete, range(1, 5));
    }

    #[test]
    fn test_coalesce_adjacent_series() {
        let mut builder = TextEditBuilder::default();
        builder.replace(range(1, 3), "au".into());
        builder.replace(range(3, 5), "www".into());
        builder.replace(range(5, 8), "".into());
        builder.replace(range(8, 9), "ub".into());

        let edit = builder.finish();
        assert_eq!(edit.indels.len(), 1);
        assert_eq!(edit.indels[0].insert, "auwwwub");
        assert_eq!(edit.indels[0].delete, range(1, 9));
    }
}
