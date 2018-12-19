use ra_text_edit::{AtomTextEdit};
use ra_syntax::{TextUnit, TextRange};
use crate::{LineIndex, LineCol};
use superslice::Ext;

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

#[derive(Debug)]
struct AltEdit<'a> {
    insert_newlines: OffsetNewlineIter<'a>,
    delete: TextRange,
    diff: i64,
}

fn translate_range_by(range: TextRange, diff: i64) -> TextRange {
    if diff == 0 {
        range
    } else {
        let start = translate_by(range.start(), diff);
        let end = translate_by(range.end(), diff);
        TextRange::from_to(start, end)
    }
}

fn translate_by(x: TextUnit, diff: i64) -> TextUnit {
    if diff == 0 {
        x
    } else {
        TextUnit::from((x.to_usize() as i64 + diff) as u32)
    }
}

fn to_alt_edits<'a>(offset: TextUnit, edits: &'a [AtomTextEdit]) -> Vec<AltEdit<'a>> {
    let mut xs: Vec<AltEdit<'a>> = Vec::with_capacity(edits.len());
    // collect and sort edits
    for edit in edits {
        // TODO discard after translating?
        // if edit.delete.start() >= offset {
        //     continue;
        // }
        let insert_index = xs.upper_bound_by_key(&edit.delete.start(), |x| x.delete.start());
        let diff = edit.insert.len() as i64 - edit.delete.len().to_usize() as i64;
        xs.insert(
            insert_index,
            AltEdit {
                insert_newlines: OffsetNewlineIter {
                    offset: edit.delete.start(),
                    text: &edit.insert,
                },
                delete: edit.delete,
                diff: diff,
            },
        );
    }
    // translate edits by previous edits
    for i in 1..xs.len() {
        let (x, prevs) = xs[0..=i].split_last_mut().unwrap();
        for prev in prevs {
            x.delete = translate_range_by(x.delete, prev.diff);
            x.insert_newlines.offset = translate_by(x.insert_newlines.offset, prev.diff);
        }
    }
    xs
}

#[derive(Debug)]
enum NextNewline {
    Use,
    Discard,
    Replace(TextUnit),
    New(TextUnit),
}

fn next_newline(candidate: Option<TextUnit>, edits: &mut [AltEdit]) -> NextNewline {
    let mut candidate = match candidate {
        None => {
            for edit in edits {
                if let Some(inserted) = edit.insert_newlines.next() {
                    return NextNewline::New(inserted);
                }
            }
            return NextNewline::Use; // END
        }
        Some(x) => x,
    };

    for edit in edits {
        if candidate <= edit.delete.start() {
            return NextNewline::Replace(candidate);
        } else if candidate <= edit.delete.end() {
            return match edit.insert_newlines.next() {
                Some(x) => NextNewline::Replace(x),
                None => NextNewline::Discard,
            };
        } else {
            if let Some(inserted) = edit.insert_newlines.next() {
                return NextNewline::New(inserted);
            }
            candidate = translate_by(candidate, edit.diff);
        }
    }
    return NextNewline::Replace(candidate);
}

fn count_newlines(offset: TextUnit, line_index: &LineIndex, edits: &[AtomTextEdit]) -> u32 {
    let mut edits = to_alt_edits(offset, edits);
    let mut orig_newlines = line_index.newlines().iter().map(|x| *x).peekable();

    let mut count = 0;

    loop {
        let res = next_newline(orig_newlines.peek().map(|x| *x), &mut edits);
        let next = match res {
            NextNewline::Use => orig_newlines.next(),
            NextNewline::Discard => {
                orig_newlines.next();
                continue;
            }
            NextNewline::Replace(new) => {
                orig_newlines.next();
                Some(new)
            }
            NextNewline::New(new) => Some(new),
        };
        match next {
            Some(n) if n <= offset => {
                count += 1;
            }
            _ => {
                break;
            }
        }
    }

    count
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

        #[test]
        fn test_translate_offset_with_edit_alt(x in arb_text_with_offset_and_edits()) {
            let line_index = LineIndex::new(&x.text);
            let expected = translate_after_edit(&x.text, x.offset, x.edits.clone());
            let actual_lines = count_newlines(x.offset, &line_index, &x.edits);
            assert_eq!(actual_lines, expected.line);
        }
    }
}
