extern crate libeditor;
extern crate itertools;

use std::fmt;
use itertools::Itertools;
use libeditor::{File, TextRange};

#[test]
fn test_extend_selection() {
    let file = file(r#"fn foo() {
    1 + 1
}
"#);
    let range = TextRange::offset_len(18.into(), 0.into());
    let range = file.extend_selection(range).unwrap();
    assert_eq!(range, TextRange::from_to(17.into(), 18.into()));
    let range = file.extend_selection(range).unwrap();
    assert_eq!(range, TextRange::from_to(15.into(), 20.into()));
}

#[test]
fn test_highlighting() {
    let file = file(r#"
// comment
fn main() {}
    println!("Hello, {}!", 92);
"#);
    let hls = file.highlight();
    dbg_eq(
        &hls,
        r#"[HighlightedRange { range: [1; 11), tag: "comment" },
            HighlightedRange { range: [12; 14), tag: "keyword" },
            HighlightedRange { range: [15; 19), tag: "function" },
            HighlightedRange { range: [29; 36), tag: "text" },
            HighlightedRange { range: [38; 50), tag: "string" },
            HighlightedRange { range: [52; 54), tag: "literal" }]"#
    );
}

fn file(text: &str) -> File {
    File::new(text)
}

fn dbg_eq(actual: &impl fmt::Debug, expected: &str) {
    let expected = expected.lines().map(|l| l.trim()).join(" ");
    assert_eq!(actual, expected);
}
