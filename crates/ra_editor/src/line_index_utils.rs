use ra_text_edit::{AtomTextEdit};
use ra_syntax::{TextUnit, TextRange};
use crate::{LineIndex, LineCol};

#[derive(Debug)]
struct OffsetNewlineIter<'a> {
    text: &'a str,
    offset: TextUnit,
}

impl<'a> Iterator for OffsetNewlineIter<'a> {
    type Item = TextUnit;
    fn next(&mut self) -> Option<TextUnit> {
        let next_idx = self
            .text
            .char_indices()
            .filter_map(|(i, c)| if c == '\n' { Some(i + 1) } else { None })
            .next()?;
        let next = self.offset + TextUnit::from_usize(next_idx);
        self.text = &self.text[next_idx..];
        self.offset = next;
        Some(next)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TranslatedPos {
    Before,
    After,
}

/// None means it was deleted
type TranslatedOffset = Option<(TranslatedPos, TextUnit)>;

fn translate_offset(offset: TextUnit, edit: &TranslatedAtomEdit) -> TranslatedOffset {
    if offset <= edit.delete.start() {
        Some((TranslatedPos::Before, offset))
    } else if offset <= edit.delete.end() {
        None
    } else {
        let diff = edit.insert.len() as i64 - edit.delete.len().to_usize() as i64;
        let after = TextUnit::from((offset.to_usize() as i64 + diff) as u32);
        Some((TranslatedPos::After, after))
    }
}

trait TranslatedNewlineIterator {
    fn translate(&self, offset: TextUnit) -> TextUnit;
    fn translate_range(&self, range: TextRange) -> TextRange {
        TextRange::from_to(self.translate(range.start()), self.translate(range.end()))
    }
    fn next_translated(&mut self) -> Option<TextUnit>;
    fn boxed<'a>(self) -> Box<TranslatedNewlineIterator + 'a>
    where
        Self: 'a + Sized,
    {
        Box::new(self)
    }
}

struct TranslatedAtomEdit<'a> {
    delete: TextRange,
    insert: &'a str,
}

struct TranslatedNewlines<'a, T: TranslatedNewlineIterator> {
    inner: T,
    next_inner: Option<TranslatedOffset>,
    edit: TranslatedAtomEdit<'a>,
    insert: OffsetNewlineIter<'a>,
}

impl<'a, T: TranslatedNewlineIterator> TranslatedNewlines<'a, T> {
    fn from(inner: T, edit: &'a AtomTextEdit) -> Self {
        let delete = inner.translate_range(edit.delete);
        let mut res = TranslatedNewlines {
            inner,
            next_inner: None,
            edit: TranslatedAtomEdit {
                delete,
                insert: &edit.insert,
            },
            insert: OffsetNewlineIter {
                offset: delete.start(),
                text: &edit.insert,
            },
        };
        // prepare next_inner
        res.advance_inner();
        res
    }

    fn advance_inner(&mut self) {
        self.next_inner = self
            .inner
            .next_translated()
            .map(|x| translate_offset(x, &self.edit));
    }
}

impl<'a, T: TranslatedNewlineIterator> TranslatedNewlineIterator for TranslatedNewlines<'a, T> {
    fn translate(&self, offset: TextUnit) -> TextUnit {
        let offset = self.inner.translate(offset);
        let (_, offset) =
            translate_offset(offset, &self.edit).expect("translate_unit returned None");
        offset
    }

    fn next_translated(&mut self) -> Option<TextUnit> {
        match self.next_inner {
            None => self.insert.next(),
            Some(next) => match next {
                None => self.insert.next().or_else(|| {
                    self.advance_inner();
                    self.next_translated()
                }),
                Some((TranslatedPos::Before, next)) => {
                    self.advance_inner();
                    Some(next)
                }
                Some((TranslatedPos::After, next)) => self.insert.next().or_else(|| {
                    self.advance_inner();
                    Some(next)
                }),
            },
        }
    }
}

impl<'a> Iterator for Box<dyn TranslatedNewlineIterator + 'a> {
    type Item = TextUnit;
    fn next(&mut self) -> Option<TextUnit> {
        self.next_translated()
    }
}

impl<T: TranslatedNewlineIterator + ?Sized> TranslatedNewlineIterator for Box<T> {
    fn translate(&self, offset: TextUnit) -> TextUnit {
        self.as_ref().translate(offset)
    }
    fn next_translated(&mut self) -> Option<TextUnit> {
        self.as_mut().next_translated()
    }
}

struct IteratorWrapper<T: Iterator<Item = TextUnit>>(T);

impl<T: Iterator<Item = TextUnit>> TranslatedNewlineIterator for IteratorWrapper<T> {
    fn translate(&self, offset: TextUnit) -> TextUnit {
        offset
    }
    fn next_translated(&mut self) -> Option<TextUnit> {
        self.0.next()
    }
}

impl<T: Iterator<Item = TextUnit>> Iterator for IteratorWrapper<T> {
    type Item = TextUnit;
    fn next(&mut self) -> Option<TextUnit> {
        self.0.next()
    }
}

fn translate_newlines<'a>(
    mut newlines: Box<TranslatedNewlineIterator + 'a>,
    edits: &'a [AtomTextEdit],
) -> Box<TranslatedNewlineIterator + 'a> {
    for edit in edits {
        newlines = TranslatedNewlines::from(newlines, edit).boxed();
    }
    newlines
}

pub fn translate_offset_with_edit(
    pre_edit_index: &LineIndex,
    offset: TextUnit,
    edits: &[AtomTextEdit],
) -> LineCol {
    let mut newlines: Box<TranslatedNewlineIterator> = Box::new(IteratorWrapper(
        pre_edit_index.newlines().iter().map(|x| *x),
    ));

    newlines = translate_newlines(newlines, edits);

    let mut line = 0;
    for n in newlines {
        if n > offset {
            break;
        }
        line += 1;
    }

    LineCol {
        line: line,
        col_utf16: 0, // TODO not implemented yet
    }
}

#[cfg(test)]
mod test {
    use proptest::{prelude::*, proptest, proptest_helper};
    use super::*;
    use ra_text_edit::test_utils::{arb_text, arb_offset, arb_edits};

    #[derive(Debug)]
    struct ArbTextWithOffsetAndEdits {
        text: String,
        offset: TextUnit,
        edits: Vec<AtomTextEdit>,
    }

    fn arb_text_with_offset_and_edits() -> BoxedStrategy<ArbTextWithOffsetAndEdits> {
        arb_text()
            .prop_flat_map(|text| {
                (arb_offset(&text), arb_edits(&text), Just(text)).prop_map(
                    |(offset, edits, text)| ArbTextWithOffsetAndEdits {
                        text,
                        offset,
                        edits,
                    },
                )
            })
            .boxed()
    }

    fn edit_text(pre_edit_text: &str, mut edits: Vec<AtomTextEdit>) -> String {
        // apply edits ordered from last to first
        // since they should not overlap we can just use start()
        edits.sort_by_key(|x| -(x.delete.start().to_usize() as isize));

        let mut text = pre_edit_text.to_owned();

        for edit in &edits {
            let range = edit.delete.start().to_usize()..edit.delete.end().to_usize();
            text.replace_range(range, &edit.insert);
        }

        text
    }

    fn translate_after_edit(
        pre_edit_text: &str,
        offset: TextUnit,
        edits: Vec<AtomTextEdit>,
    ) -> LineCol {
        let text = edit_text(pre_edit_text, edits);
        let line_index = LineIndex::new(&text);
        line_index.line_col(offset)
    }

    proptest! {
        #[test]
        fn test_translate_offset_with_edit(x in arb_text_with_offset_and_edits()) {
            let line_index = LineIndex::new(&x.text);
            let expected = translate_after_edit(&x.text, x.offset, x.edits.clone());
            let actual = translate_offset_with_edit(&line_index, x.offset, &x.edits);
            assert_eq!(actual.line, expected.line);
        }
    }
}
