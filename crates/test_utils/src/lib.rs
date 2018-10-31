extern crate difference;
extern crate itertools;
extern crate text_unit;

use itertools::Itertools;
use std::fmt;
use text_unit::{TextRange, TextUnit};

pub use self::difference::Changeset as __Changeset;

pub const CURSOR_MARKER: &str = "<|>";

#[macro_export]
macro_rules! assert_eq_text {
    ($expected:expr, $actual:expr) => {{
        let expected = $expected;
        let actual = $actual;
        if expected != actual {
            let changeset = $crate::__Changeset::new(actual, expected, "\n");
            println!("Expected:\n{}\n\nActual:\n{}\nDiff:{}\n", expected, actual, changeset);
            panic!("text differs");
        }
    }};
    ($expected:expr, $actual:expr, $($tt:tt)*) => {{
        let expected = $expected;
        let actual = $actual;
        if expected != actual {
            let changeset = $crate::__Changeset::new(actual, expected, "\n");
            println!("Expected:\n{}\n\nActual:\n{}\n\nDiff:\n{}\n", expected, actual, changeset);
            println!($($tt)*);
            panic!("text differs");
        }
    }};
}

pub fn assert_eq_dbg(expected: &str, actual: &impl fmt::Debug) {
    let actual = format!("{:?}", actual);
    let expected = expected.lines().map(|l| l.trim()).join(" ");
    assert_eq!(expected, actual);
}

pub fn extract_offset(text: &str) -> (TextUnit, String) {
    match try_extract_offset(text) {
        None => panic!("text should contain cursor marker"),
        Some(result) => result,
    }
}

pub fn try_extract_offset(text: &str) -> Option<(TextUnit, String)> {
    let cursor_pos = text.find(CURSOR_MARKER)?;
    let mut new_text = String::with_capacity(text.len() - CURSOR_MARKER.len());
    new_text.push_str(&text[..cursor_pos]);
    new_text.push_str(&text[cursor_pos + CURSOR_MARKER.len()..]);
    let cursor_pos = TextUnit::from(cursor_pos as u32);
    Some((cursor_pos, new_text))
}

pub fn extract_range(text: &str) -> (TextRange, String) {
    match try_extract_range(text) {
        None => panic!("text should contain cursor marker"),
        Some(result) => result,
    }
}

pub fn try_extract_range(text: &str) -> Option<(TextRange, String)> {
    let (start, text) = try_extract_offset(text)?;
    let (end, text) = try_extract_offset(&text)?;
    Some((TextRange::from_to(start, end), text))
}

pub fn extract_ranges(text: &str) -> (Vec<TextRange>, String) {
    let mut ranges = Vec::new();
    let mut text = String::from(text);
    while let Some((range, new_text)) = try_extract_range(&text) {
        text = new_text;
        ranges.push(range);
    }

    (ranges, text)
}

pub fn add_cursor(text: &str, offset: TextUnit) -> String {
    let offset: u32 = offset.into();
    let offset: usize = offset as usize;
    let mut res = String::new();
    res.push_str(&text[..offset]);
    res.push_str("<|>");
    res.push_str(&text[offset..]);
    res
}

#[derive(Debug)]
pub struct FixtureEntry {
    pub meta: String,
    pub text: String,
}

/// Parses text wich looks like this:
///
///  ```notrust
///  //- some meta
///  line 1
///  line 2
///  // - other meta
///  ```
pub fn parse_fixture(fixture: &str) -> Vec<FixtureEntry> {
    let mut res = Vec::new();
    let mut buf = String::new();
    let mut meta: Option<&str> = None;

    macro_rules! flush {
        () => {
            if let Some(meta) = meta {
                res.push(FixtureEntry {
                    meta: meta.to_string(),
                    text: buf.clone(),
                });
                buf.clear();
            }
        };
    };
    let margin = fixture
        .lines()
        .filter(|it| it.trim_start().starts_with("//-"))
        .map(|it| it.len() - it.trim_start().len())
        .next()
        .expect("empty fixture");
    let lines = fixture.lines().filter_map(|line| {
        if line.len() >= margin {
            assert!(line[..margin].trim().is_empty());
            Some(&line[margin..])
        } else {
            assert!(line.trim().is_empty());
            None
        }
    });

    for line in lines {
        if line.starts_with("//-") {
            flush!();
            buf.clear();
            meta = Some(line["//-".len()..].trim());
            continue;
        }
        buf.push_str(line);
        buf.push('\n');
    }
    flush!();
    res
}
