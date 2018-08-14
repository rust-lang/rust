extern crate libeditor;
extern crate libsyntax2;
extern crate itertools;
#[macro_use]
extern crate assert_eq_text;

use std::fmt;
use itertools::Itertools;
use libsyntax2::AstNode;
use libeditor::{
    File, TextUnit, TextRange,
    highlight, runnables, extend_selection, file_structure, flip_comma,
};

#[test]
fn test_extend_selection() {
    let file = file(r#"fn foo() {
    1 + 1
}
"#);
    let range = TextRange::offset_len(18.into(), 0.into());
    let range = extend_selection(&file, range).unwrap();
    assert_eq!(range, TextRange::from_to(17.into(), 18.into()));
    let range = extend_selection(&file, range).unwrap();
    assert_eq!(range, TextRange::from_to(15.into(), 20.into()));
}

#[test]
fn test_highlighting() {
    let file = file(r#"
// comment
fn main() {}
    println!("Hello, {}!", 92);
"#);
    let hls = highlight(&file);
    dbg_eq(
        r#"[HighlightedRange { range: [1; 11), tag: "comment" },
            HighlightedRange { range: [12; 14), tag: "keyword" },
            HighlightedRange { range: [15; 19), tag: "function" },
            HighlightedRange { range: [29; 36), tag: "text" },
            HighlightedRange { range: [38; 50), tag: "string" },
            HighlightedRange { range: [52; 54), tag: "literal" }]"#,
        &hls,
    );
}

#[test]
fn test_runnables() {
    let file = file(r#"
fn main() {}

#[test]
fn test_foo() {}

#[test]
#[ignore]
fn test_foo() {}
"#);
    let runnables = runnables(&file);
    dbg_eq(
        r#"[Runnable { range: [1; 13), kind: Bin },
            Runnable { range: [15; 39), kind: Test { name: "test_foo" } },
            Runnable { range: [41; 75), kind: Test { name: "test_foo" } }]"#,
        &runnables,
    )
}

#[test]
fn test_file_structure() {
    let file = file(r#"
struct Foo {
    x: i32
}

mod m {
    fn bar() {}
}

enum E { X, Y(i32) }
type T = ();
static S: i32 = 92;
const C: i32 = 92;

impl E {}

impl fmt::Debug for E {}
"#);
    let symbols = file_structure(&file);
    dbg_eq(
        r#"[StructureNode { parent: None, label: "Foo", navigation_range: [8; 11), node_range: [1; 26), kind: STRUCT_DEF },
           StructureNode { parent: None, label: "m", navigation_range: [32; 33), node_range: [28; 53), kind: MODULE },
           StructureNode { parent: Some(1), label: "bar", navigation_range: [43; 46), node_range: [40; 51), kind: FN_DEF },
           StructureNode { parent: None, label: "E", navigation_range: [60; 61), node_range: [55; 75), kind: ENUM_DEF },
           StructureNode { parent: None, label: "T", navigation_range: [81; 82), node_range: [76; 88), kind: TYPE_DEF },
           StructureNode { parent: None, label: "S", navigation_range: [96; 97), node_range: [89; 108), kind: STATIC_DEF },
           StructureNode { parent: None, label: "C", navigation_range: [115; 116), node_range: [109; 127), kind: CONST_DEF },
           StructureNode { parent: None, label: "impl E", navigation_range: [134; 135), node_range: [129; 138), kind: IMPL_ITEM },
           StructureNode { parent: None, label: "impl fmt::Debug for E", navigation_range: [160; 161), node_range: [140; 164), kind: IMPL_ITEM }]"#,
        &symbols,
    )
}

#[test]
fn test_swap_comma() {
    check_modification(
        "fn foo(x: i32,<|> y: Result<(), ()>) {}",
        "fn foo(y: Result<(), ()>, x: i32) {}",
        &|file, offset| {
            let edit = flip_comma(file, offset).unwrap()();
            edit.apply(&file.syntax().text())
        },
    )
}

fn file(text: &str) -> File {
    File::parse(text)
}

fn dbg_eq(expected: &str, actual: &impl fmt::Debug) {
    let actual = format!("{:?}", actual);
    let expected = expected.lines().map(|l| l.trim()).join(" ");
    assert_eq!(expected, actual);
}

fn check_modification(
    before: &str,
    after: &str,
    f: &impl Fn(&File, TextUnit) -> String,
) {
    let cursor = "<|>";
    let cursor_pos = match before.find(cursor) {
        None => panic!("before text should contain cursor marker"),
        Some(pos) => pos,
    };
    let mut text = String::with_capacity(before.len() - cursor.len());
    text.push_str(&before[..cursor_pos]);
    text.push_str(&before[cursor_pos + cursor.len()..]);
    let cursor_pos = TextUnit::from(cursor_pos as u32);
    let file = file(&text);
    let actual = f(&file, cursor_pos);
    assert_eq_text!(after, &actual);
}
