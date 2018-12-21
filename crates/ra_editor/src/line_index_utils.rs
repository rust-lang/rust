use ra_text_edit::AtomTextEdit;
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

    fn next_newlines(&mut self, candidate: TextUnit) -> NextNewlines {
        let res = match &mut self.current {
            Some(edit) => {
                if candidate <= edit.delete.start() {
                    NextNewlines::Use
                } else if candidate <= edit.delete.end() {
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
}

pub fn count_newlines(offset: TextUnit, line_index: &LineIndex, edits: &[AtomTextEdit]) -> u32 {
    let mut sorted_edits: Vec<&AtomTextEdit> = Vec::with_capacity(edits.len());
    for edit in edits {
        let insert_index =
            sorted_edits.upper_bound_by_key(&edit.delete.start(), |x| x.delete.start());
        sorted_edits.insert(insert_index, &edit);
    }

    let mut state = Edits::new(&sorted_edits);

    let mut lines: u32 = 0;

    for &orig_newline in line_index.newlines() {
        loop {
            let translated_newline = state.translate(orig_newline);
            match state.next_newlines(translated_newline) {
                NextNewlines::Use => {
                    if offset < translated_newline {
                        return lines;
                    } else {
                        lines += 1;
                    }
                    break;
                }
                NextNewlines::ReplaceMany(ns) => {
                    for n in ns {
                        if offset < n {
                            return lines;
                        } else {
                            lines += 1;
                        }
                    }
                    break;
                }
                NextNewlines::AddMany(ns) => {
                    for n in ns {
                        if offset < n {
                            return lines;
                        } else {
                            lines += 1;
                        }
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
                    if offset < n {
                        return lines;
                    } else {
                        lines += 1;
                    }
                }
            }
        }
    }

    lines
}

// for bench
pub fn translate_after_edit(
    pre_edit_text: &str,
    offset: TextUnit,
    edits: Vec<AtomTextEdit>,
) -> LineCol {
    let text = edit_text(pre_edit_text, edits);
    let line_index = LineIndex::new(&text);
    line_index.line_col(offset)
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

    proptest! {
        #[test]
        fn test_translate_offset_with_edit(x in arb_text_with_offset_and_edits()) {
            let line_index = LineIndex::new(&x.text);
            let expected = translate_after_edit(&x.text, x.offset, x.edits.clone());
            let actual_lines = count_newlines(x.offset, &line_index, &x.edits);
            assert_eq!(actual_lines, expected.line);
        }
    }
}
