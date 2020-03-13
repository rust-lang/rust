//! Code actions can specify desirable final position of the cursor.
//!
//! The position is specified as a `TextUnit` in the final file. We need to send
//! it in `(Line, Column)` coordinate though. However, we only have a LineIndex
//! for a file pre-edit!
//!
//! Code in this module applies this "to (Line, Column) after edit"
//! transformation.

use ra_syntax::{TextRange, TextUnit};
use ra_text_edit::{AtomTextEdit, TextEdit};

use crate::line_index::{LineCol, LineIndex, Utf16Char};

pub fn translate_offset_with_edit(
    line_index: &LineIndex,
    offset: TextUnit,
    text_edit: &TextEdit,
) -> LineCol {
    let mut state = Edits::from_text_edit(&text_edit);

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
                        res.adjust_col(*x);
                    }
                }
            }
        };
    }

    for orig_step in LineIndexStepIter::from(line_index) {
        loop {
            let translated_step = state.translate_step(&orig_step);
            match state.next_steps(&translated_step) {
                NextSteps::Use => {
                    test_step!(translated_step);
                    break;
                }
                NextSteps::ReplaceMany(ns) => {
                    for n in ns {
                        test_step!(n);
                    }
                    break;
                }
                NextSteps::AddMany(ns) => {
                    for n in ns {
                        test_step!(n);
                    }
                }
            }
        }
    }

    loop {
        match state.next_inserted_steps() {
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

impl LineIndexStepIter<'_> {
    fn from(line_index: &LineIndex) -> LineIndexStepIter {
        let mut x = LineIndexStepIter { line_index, next_newline_idx: 0, utf16_chars: None };
        // skip first newline since it's not real
        x.next();
        x
    }
}

impl Iterator for LineIndexStepIter<'_> {
    type Item = Step;
    fn next(&mut self) -> Option<Step> {
        self.utf16_chars
            .as_mut()
            .and_then(|(newline, x)| {
                let x = x.next()?;
                Some(Step::Utf16Char(TextRange::from_to(*newline + x.start, *newline + x.end)))
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
struct OffsetStepIter<'a> {
    text: &'a str,
    offset: TextUnit,
}

impl Iterator for OffsetStepIter<'_> {
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
                    if char_len > TextUnit::from_usize(1) {
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
enum NextSteps<'a> {
    Use,
    ReplaceMany(OffsetStepIter<'a>),
    AddMany(OffsetStepIter<'a>),
}

#[derive(Debug)]
struct TranslatedEdit<'a> {
    delete: TextRange,
    insert: &'a str,
    diff: i64,
}

struct Edits<'a> {
    edits: &'a [AtomTextEdit],
    current: Option<TranslatedEdit<'a>>,
    acc_diff: i64,
}

impl<'a> Edits<'a> {
    fn from_text_edit(text_edit: &'a TextEdit) -> Edits<'a> {
        let mut x = Edits { edits: text_edit.as_atoms(), current: None, acc_diff: 0 };
        x.advance_edit();
        x
    }
    fn advance_edit(&mut self) {
        self.acc_diff += self.current.as_ref().map_or(0, |x| x.diff);
        match self.edits.split_first() {
            Some((next, rest)) => {
                let delete = self.translate_range(next.delete);
                let diff = next.insert.len() as i64 - next.delete.len().to_usize() as i64;
                self.current = Some(TranslatedEdit { delete, insert: &next.insert, diff });
                self.edits = rest;
            }
            None => {
                self.current = None;
            }
        }
    }

    fn next_inserted_steps(&mut self) -> Option<OffsetStepIter<'a>> {
        let cur = self.current.as_ref()?;
        let res = Some(OffsetStepIter { offset: cur.delete.start(), text: &cur.insert });
        self.advance_edit();
        res
    }

    fn next_steps(&mut self, step: &Step) -> NextSteps {
        let step_pos = match *step {
            Step::Newline(n) => n,
            Step::Utf16Char(r) => r.end(),
        };
        match &mut self.current {
            Some(edit) => {
                if step_pos <= edit.delete.start() {
                    NextSteps::Use
                } else if step_pos <= edit.delete.end() {
                    let iter = OffsetStepIter { offset: edit.delete.start(), text: &edit.insert };
                    // empty slice to avoid returning steps again
                    edit.insert = &edit.insert[edit.insert.len()..];
                    NextSteps::ReplaceMany(iter)
                } else {
                    let iter = OffsetStepIter { offset: edit.delete.start(), text: &edit.insert };
                    // empty slice to avoid returning steps again
                    edit.insert = &edit.insert[edit.insert.len()..];
                    self.advance_edit();
                    NextSteps::AddMany(iter)
                }
            }
            None => NextSteps::Use,
        }
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
            match *x {
                Step::Newline(n) => Step::Newline(self.translate(n)),
                Step::Utf16Char(r) => Step::Utf16Char(self.translate_range(r)),
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
        RunningLineCol { line: 0, last_newline: TextUnit::from(0), col_adjust: TextUnit::from(0) }
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

    fn adjust_col(&mut self, range: TextRange) {
        self.col_adjust += range.len() - TextUnit::from(1);
    }
}
