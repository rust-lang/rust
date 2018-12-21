use proptest::prelude::*;
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
    arb_edits_custom(&text, 0, 7)
}

pub fn arb_edits_custom(text: &str, min: usize, max: usize) -> BoxedStrategy<Vec<AtomTextEdit>> {
    if text.is_empty() {
        // only valid edits
        return Just(vec![])
            .boxed()
            .prop_union(
                arb_text()
                    .prop_map(|text| vec![AtomTextEdit::insert(TextUnit::from(0), text)])
                    .boxed(),
            )
            .boxed();
    }

    let offsets = text_offsets(text);
    let max_cuts = max.min(offsets.len());
    let min_cuts = min.min(offsets.len() - 1);

    proptest::sample::subsequence(offsets, min_cuts..max_cuts)
        .prop_flat_map(|cuts| {
            let strategies: Vec<_> = cuts
                .chunks(2)
                .map(|chunk| match chunk {
                    &[from, to] => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{proptest, proptest_helper};

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
        for i in 1..edits.len() {
            let e1 = &edits[i];
            for e2 in &edits[0..i] {
                if intersect(e1.delete, e2.delete).is_some() {
                    assert!(false, "Overlapping ranges {} {}", e1.delete, e2.delete);
                }
            }
        }
    }
}
