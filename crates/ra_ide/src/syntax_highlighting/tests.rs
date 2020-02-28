use std::fs;

use test_utils::{assert_eq_text, project_dir, read_text};

use crate::{
    mock_analysis::{single_file, MockAnalysis},
    FileRange, TextRange,
};

#[test]
fn test_highlighting() {
    let (analysis, file_id) = single_file(
        r#"
#[derive(Clone, Debug)]
struct Foo {
    pub x: i32,
    pub y: i32,
}

fn foo<T>() -> T {
    unimplemented!();
    foo::<i32>();
}

macro_rules! def_fn {
    ($($tt:tt)*) => {$($tt)*}
}

def_fn! {
    fn bar() -> u32 {
        100
    }
}

// comment
fn main() {
    println!("Hello, {}!", 92);

    let mut vec = Vec::new();
    if true {
        let x = 92;
        vec.push(Foo { x, y: 1 });
    }
    unsafe { vec.set_len(0); }

    let mut x = 42;
    let y = &mut x;
    let z = &y;

    y;
}

enum Option<T> {
    Some(T),
    None,
}
use Option::*;

impl<T> Option<T> {
    fn and<U>(self, other: Option<U>) -> Option<(T, U)> {
        match other {
            None => todo!(),
            Nope => Nope,
        }
    }
}
"#
        .trim(),
    );
    let dst_file = project_dir().join("crates/ra_ide/src/snapshots/highlighting.html");
    let actual_html = &analysis.highlight_as_html(file_id, false).unwrap();
    let expected_html = &read_text(&dst_file);
    fs::write(dst_file, &actual_html).unwrap();
    assert_eq_text!(expected_html, actual_html);
}

#[test]
fn test_rainbow_highlighting() {
    let (analysis, file_id) = single_file(
        r#"
fn main() {
    let hello = "hello";
    let x = hello.to_string();
    let y = hello.to_string();

    let x = "other color please!";
    let y = x.to_string();
}

fn bar() {
    let mut hello = "hello";
}
"#
        .trim(),
    );
    let dst_file = project_dir().join("crates/ra_ide/src/snapshots/rainbow_highlighting.html");
    let actual_html = &analysis.highlight_as_html(file_id, true).unwrap();
    let expected_html = &read_text(&dst_file);
    fs::write(dst_file, &actual_html).unwrap();
    assert_eq_text!(expected_html, actual_html);
}

#[test]
fn accidentally_quadratic() {
    let file = project_dir().join("crates/ra_syntax/test_data/accidentally_quadratic");
    let src = fs::read_to_string(file).unwrap();

    let mut mock = MockAnalysis::new();
    let file_id = mock.add_file("/main.rs", &src);
    let host = mock.analysis_host();

    // let t = std::time::Instant::now();
    let _ = host.analysis().highlight(file_id).unwrap();
    // eprintln!("elapsed: {:?}", t.elapsed());
}

#[test]
fn test_ranges() {
    let (analysis, file_id) = single_file(
        r#"
            #[derive(Clone, Debug)]
            struct Foo {
                pub x: i32,
                pub y: i32,
            }"#,
    );

    // The "x"
    let highlights = &analysis
        .highlight_range(FileRange { file_id, range: TextRange::offset_len(82.into(), 1.into()) })
        .unwrap();

    assert_eq!(&highlights[0].highlight.to_string(), "field");
}
