use ra_text_edit::AtomTextEdit;
use ra_syntax::{TextUnit, TextRange};
use crate::{LineIndex, LineCol, line_index::Utf16Char, line_index};
use superslice::Ext;

#[derive(Debug, Clone)]
enum Step {
    Newline(TextUnit),
    Utf16Char(TextRange),
}

#[derive(Debug)]
struct LineIndexStepIter<'a> {
    line_index: &'a LineIndex,
    next_newline_idx: usize,
    utf16_chars: Option<(TextUnit, std::slice::Iter<'a, Utf16Char>)>,
}

impl<'a> LineIndexStepIter<'a> {
    fn from(line_index: &LineIndex) -> LineIndexStepIter {
        let mut x = LineIndexStepIter {
            line_index,
            next_newline_idx: 0,
            utf16_chars: None,
        };
        // skip first newline since it's not real
        x.next();
        x
    }
}

impl<'a> Iterator for LineIndexStepIter<'a> {
    type Item = Step;
    fn next(&mut self) -> Option<Step> {
        self.utf16_chars
            .as_mut()
            .and_then(|(newline, x)| {
                let x = x.next()?;
                Some(Step::Utf16Char(TextRange::from_to(
                    *newline + x.start,
                    *newline + x.end,
                )))
            })
            .or_else(|| {
                let next_newline = *self.line_index.newlines.get(self.next_newline_idx)?;
                self.utf16_chars = self
                    .line_index
                    .utf16_lines
                    .get(&(self.next_newline_idx as u32))
                    .map(|x| (next_newline, x.iter()));
                self.next_newline_idx += 1;
                Some(Step::Newline(next_newline))
            })
    }
}

#[derive(Debug)]
struct OffsetNewlineIter<'a> {
    text: &'a str,
    offset: TextUnit,
}

impl<'a> Iterator for OffsetNewlineIter<'a> {
    type Item = Step;
    fn next(&mut self) -> Option<Step> {
        let (next, next_offset) = self
            .text
            .char_indices()
            .filter_map(|(i, c)| {
                if c == '\n' {
                    let next_offset = self.offset + TextUnit::from_usize(i + 1);
                    let next = Step::Newline(next_offset);
                    Some((next, next_offset))
                } else {
                    let char_len = TextUnit::of_char(c);
                    if char_len.to_usize() > 1 {
                        let start = self.offset + TextUnit::from_usize(i);
                        let end = start + char_len;
                        let next = Step::Utf16Char(TextRange::from_to(start, end));
                        let next_offset = end;
                        Some((next, next_offset))
                    } else {
                        None
                    }
                }
            })
            .next()?;
        let next_idx = (next_offset - self.offset).to_usize();
        self.text = &self.text[next_idx..];
        self.offset = next_offset;
        Some(next)
    }
}

#[derive(Debug)]
enum NextNewlines<'a> {
    Use,
    ReplaceMany(OffsetNewlineIter<'a>),
    AddMany(OffsetNewlineIter<'a>),
}

#[derive(Debug)]
struct TranslatedEdit<'a> {
    delete: TextRange,
    insert: &'a str,
    diff: i64,
}

struct Edits<'a, 'b> {
    edits: &'b [&'a AtomTextEdit],
    current: Option<TranslatedEdit<'a>>,
    acc_diff: i64,
}

impl<'a, 'b> Edits<'a, 'b> {
    fn new(sorted_edits: &'b [&'a AtomTextEdit]) -> Edits<'a, 'b> {
        let mut x = Edits {
            edits: sorted_edits,
            current: None,
            acc_diff: 0,
        };
        x.advance_edit();
        x
    }
    fn advance_edit(&mut self) {
        self.acc_diff += self.current.as_ref().map_or(0, |x| x.diff);
        match self.edits.split_first() {
            Some((next, rest)) => {
                let delete = self.translate_range(next.delete);
                let diff = next.insert.len() as i64 - next.delete.len().to_usize() as i64;
                self.current = Some(TranslatedEdit {
                    delete,
                    insert: &next.insert,
                    diff,
                });
                self.edits = rest;
            }
            None => {
                self.current = None;
            }
        }
    }

    fn next_inserted_newlines(&mut self) -> Option<OffsetNewlineIter<'a>> {
        let cur = self.current.as_ref()?;
        let res = Some(OffsetNewlineIter {
            offset: cur.delete.start(),
            text: &cur.insert,
        });
        self.advance_edit();
        res
    }

    fn next_step(&mut self, step: &Step) -> NextNewlines {
        let step_pos = match step {
            &Step::Newline(n) => n,
            &Step::Utf16Char(r) => r.end(),
        };
        let res = match &mut self.current {
            Some(edit) => {
                if step_pos <= edit.delete.start() {
                    NextNewlines::Use
                } else if step_pos <= edit.delete.end() {
                    let iter = OffsetNewlineIter {
                        offset: edit.delete.start(),
                        text: &edit.insert,
                    };
                    // empty slice
                    edit.insert = &edit.insert[edit.insert.len()..];
                    NextNewlines::ReplaceMany(iter)
                } else {
                    let iter = OffsetNewlineIter {
                        offset: edit.delete.start(),
                        text: &edit.insert,
                    };
                    // empty slice
                    edit.insert = &edit.insert[edit.insert.len()..];
                    self.advance_edit();
                    NextNewlines::AddMany(iter)
                }
            }
            None => NextNewlines::Use,
        };
        res
    }

    fn translate_range(&self, range: TextRange) -> TextRange {
        if self.acc_diff == 0 {
            range
        } else {
            let start = self.translate(range.start());
            let end = self.translate(range.end());
            TextRange::from_to(start, end)
        }
    }

    fn translate(&self, x: TextUnit) -> TextUnit {
        if self.acc_diff == 0 {
            x
        } else {
            TextUnit::from((x.to_usize() as i64 + self.acc_diff) as u32)
        }
    }

    fn translate_step(&self, x: &Step) -> Step {
        if self.acc_diff == 0 {
            x.clone()
        } else {
            match x {
                &Step::Newline(n) => Step::Newline(self.translate(n)),
                &Step::Utf16Char(r) => Step::Utf16Char(self.translate_range(r)),
            }
        }
    }
}

#[derive(Debug)]
struct RunningLineCol {
    line: u32,
    last_newline: TextUnit,
    col_adjust: TextUnit,
}

impl RunningLineCol {
    fn new() -> RunningLineCol {
        RunningLineCol {
            line: 0,
            last_newline: TextUnit::from(0),
            col_adjust: TextUnit::from(0),
        }
    }

    fn to_line_col(&self, offset: TextUnit) -> LineCol {
        LineCol {
            line: self.line,
            col_utf16: ((offset - self.last_newline) - self.col_adjust).into(),
        }
    }

    fn add_line(&mut self, newline: TextUnit) {
        self.line += 1;
        self.last_newline = newline;
        self.col_adjust = TextUnit::from(0);
    }

    fn adjust_col(&mut self, range: &TextRange) {
        self.col_adjust += range.len() - TextUnit::from(1);
    }
}

pub fn translate_offset_with_edit(
    line_index: &LineIndex,
    offset: TextUnit,
    edits: &[AtomTextEdit],
) -> LineCol {
    let mut sorted_edits: Vec<&AtomTextEdit> = Vec::with_capacity(edits.len());
    for edit in edits {
        let insert_index =
            sorted_edits.upper_bound_by_key(&edit.delete.start(), |x| x.delete.start());
        sorted_edits.insert(insert_index, &edit);
    }

    let mut state = Edits::new(&sorted_edits);

    let mut res = RunningLineCol::new();

    macro_rules! test_step {
        ($x:ident) => {
            match &$x {
                Step::Newline(n) => {
                    if offset < *n {
                        return res.to_line_col(offset);
                    } else {
                        res.add_line(*n);
                    }
                }
                Step::Utf16Char(x) => {
                    if offset < x.end() {
                        // if the offset is inside a multibyte char it's invalid
                        // clamp it to the start of the char
                        let clamp = offset.min(x.start());
                        return res.to_line_col(clamp);
                    } else {
                        res.adjust_col(x);
                    }
                }
            }
        };
    }

    for orig_step in LineIndexStepIter::from(line_index) {
        loop {
            let translated_step = state.translate_step(&orig_step);
            match state.next_step(&translated_step) {
                NextNewlines::Use => {
                    test_step!(translated_step);
                    break;
                }
                NextNewlines::ReplaceMany(ns) => {
                    for n in ns {
                        test_step!(n);
                    }
                    break;
                }
                NextNewlines::AddMany(ns) => {
                    for n in ns {
                        test_step!(n);
                    }
                }
            }
        }
    }

    loop {
        match state.next_inserted_newlines() {
            None => break,
            Some(ns) => {
                for n in ns {
                    test_step!(n);
                }
            }
        }
    }

    res.to_line_col(offset)
}

// for bench
pub fn translate_after_edit(
    pre_edit_text: &str,
    offset: TextUnit,
    edits: Vec<AtomTextEdit>,
) -> LineCol {
    let text = edit_text(pre_edit_text, edits);
    line_index::to_line_col(&text, offset)
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

#[cfg(test)]
mod test {
    use proptest::{prelude::*, proptest, proptest_helper};
    use super::*;
    use ra_text_edit::test_utils::{arb_text, arb_offset, arb_edits};

    #[derive(Debug)]
    struct ArbTextWithOffsetAndEdits {
        text: String,
        edited_text: String,
        offset: TextUnit,
        edits: Vec<AtomTextEdit>,
    }

    fn arb_text_with_offset_and_edits() -> BoxedStrategy<ArbTextWithOffsetAndEdits> {
        arb_text()
            .prop_flat_map(|text| {
                (arb_edits(&text), Just(text)).prop_flat_map(|(edits, text)| {
                    let edited_text = edit_text(&text, edits.clone());
                    let arb_offset = arb_offset(&edited_text);
                    (Just(text), Just(edited_text), Just(edits), arb_offset).prop_map(
                        |(text, edited_text, edits, offset)| ArbTextWithOffsetAndEdits {
                            text,
                            edits,
                            edited_text,
                            offset,
                        },
                    )
                })
            })
            .boxed()
    }

    proptest! {
        #[test]
        fn test_translate_offset_with_edit(x in arb_text_with_offset_and_edits()) {
            let expected = line_index::to_line_col(&x.edited_text, x.offset);
            let line_index = LineIndex::new(&x.text);
            let actual = translate_offset_with_edit(&line_index, x.offset, &x.edits);

            assert_eq!(actual, expected);
        }
    }
}
