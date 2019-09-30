//! FIXME: write short doc here

use crate::{AtomTextEdit, TextEdit};
use proptest::prelude::*;
use text_unit::{TextRange, TextUnit};

pub fn arb_text() -> proptest::string::RegexGeneratorStrategy<String> {
    // generate multiple newlines
    proptest::string::string_regex("(.*\n?)*").unwrap()
}

fn text_offsets(text: &str) -> Vec<TextUnit> {
    text.char_indices().map(|(i, _)| TextUnit::from_usize(i)).collect()
}

pub fn arb_offset(text: &str) -> BoxedStrategy<TextUnit> {
    let offsets = text_offsets(text);
    // this is necessary to avoid "Uniform::new called with `low >= high`" panic
    if offsets.is_empty() {
        Just(TextUnit::from(0)).boxed()
    } else {
        prop::sample::select(offsets).boxed()
    }
}

pub fn arb_text_edit(text: &str) -> BoxedStrategy<TextEdit> {
    if text.is_empty() {
        // only valid edits
        return Just(vec![])
            .boxed()
            .prop_union(
                arb_text()
                    .prop_map(|text| vec![AtomTextEdit::insert(TextUnit::from(0), text)])
                    .boxed(),
            )
            .prop_map(TextEdit::from_atoms)
            .boxed();
    }

    let offsets = text_offsets(text);
    let max_cuts = 7.min(offsets.len());

    proptest::sample::subsequence(offsets, 0..max_cuts)
        .prop_flat_map(|cuts| {
            let strategies: Vec<_> = cuts
                .chunks(2)
                .map(|chunk| match *chunk {
                    [from, to] => {
                        let range = TextRange::from_to(from, to);
                        Just(AtomTextEdit::delete(range))
                            .boxed()
                            .prop_union(
                                arb_text()
                                    .prop_map(move |text| AtomTextEdit::replace(range, text))
                                    .boxed(),
                            )
                            .boxed()
                    }
                    [x] => arb_text().prop_map(move |text| AtomTextEdit::insert(x, text)).boxed(),
                    _ => unreachable!(),
                })
                .collect();
            strategies
        })
        .prop_map(TextEdit::from_atoms)
        .boxed()
}

#[derive(Debug, Clone)]
pub struct ArbTextWithEdit {
    pub text: String,
    pub edit: TextEdit,
}

pub fn arb_text_with_edit() -> BoxedStrategy<ArbTextWithEdit> {
    let text = arb_text();
    text.prop_flat_map(|s| {
        let edit = arb_text_edit(&s);
        (Just(s), edit)
    })
    .prop_map(|(text, edit)| ArbTextWithEdit { text, edit })
    .boxed()
}
