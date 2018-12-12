use proptest::{prelude::*, proptest, proptest_helper};
use text_unit::{TextUnit, TextRange};
use crate::AtomTextEdit;

pub fn arb_text() -> proptest::string::RegexGeneratorStrategy<String> {
    // generate multiple newlines
    proptest::string::string_regex("(.*\n?)*").unwrap()
}

fn text_offsets(text: &str) -> Vec<TextUnit> {
    text.char_indices()
        .map(|(i, _)| TextUnit::from_usize(i))
        .collect()
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

pub fn arb_edits(text: &str) -> BoxedStrategy<Vec<AtomTextEdit>> {
    let offsets = text_offsets(text);
    let offsets_len = offsets.len();

    if offsets_len == 0 {
        return proptest::bool::ANY
            .prop_flat_map(|b| {
                // only valid edits
                if b {
                    arb_text()
                        .prop_map(|text| vec![AtomTextEdit::insert(TextUnit::from(0), text)])
                        .boxed()
                } else {
                    Just(vec![]).boxed()
                }
            })
            .boxed();
    }

    proptest::sample::subsequence(offsets, 0..offsets_len)
        .prop_flat_map(|xs| {
            let strategies: Vec<_> = xs
                .chunks(2)
                .map(|chunk| match chunk {
                    &[from, to] => {
                        let range = TextRange::from_to(from, to);
                        (proptest::bool::ANY)
                            .prop_flat_map(move |b| {
                                if b {
                                    Just(AtomTextEdit::delete(range)).boxed()
                                } else {
                                    arb_text()
                                        .prop_map(move |text| AtomTextEdit::replace(range, text))
                                        .boxed()
                                }
                            })
                            .boxed()
                    }
                    &[x] => arb_text()
                        .prop_map(move |text| AtomTextEdit::insert(x, text))
                        .boxed(),
                    _ => unreachable!(),
                })
                .collect();
            strategies
        })
        .boxed()
}

fn arb_text_with_edits() -> BoxedStrategy<(String, Vec<AtomTextEdit>)> {
    let text = arb_text();
    text.prop_flat_map(|s| {
        let edits = arb_edits(&s);
        (Just(s), edits)
    })
    .boxed()
}

fn intersect(r1: TextRange, r2: TextRange) -> Option<TextRange> {
    let start = r1.start().max(r2.start());
    let end = r1.end().min(r2.end());
    if start <= end {
        Some(TextRange::from_to(start, end))
    } else {
        None
    }
}

proptest! {
#[test]
    fn atom_text_edits_are_valid((text, edits) in arb_text_with_edits()) {
        proptest_atom_text_edits_are_valid(text, edits)
    }
}

fn proptest_atom_text_edits_are_valid(text: String, edits: Vec<AtomTextEdit>) {
    // slicing doesn't panic
    for e in &edits {
        let _ = &text[e.delete];
    }
    // ranges do not overlap
    for (i1, e1) in edits.iter().skip(1).enumerate() {
        for e2 in &edits[0..i1] {
            if intersect(e1.delete, e2.delete).is_some() {
                assert!(false, "Overlapping ranges {} {}", e1.delete, e2.delete);
            }
        }
    }
}
