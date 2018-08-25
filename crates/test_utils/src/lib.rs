extern crate difference;
extern crate itertools;
extern crate text_unit;

use std::fmt;
use itertools::Itertools;
use text_unit::{TextUnit, TextRange};

pub use self::difference::Changeset as __Changeset;

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
    let cursor = "<|>";
    let cursor_pos = match text.find(cursor) {
        None => panic!("text should contain cursor marker"),
        Some(pos) => pos,
    };
    let mut new_text = String::with_capacity(text.len() - cursor.len());
    new_text.push_str(&text[..cursor_pos]);
    new_text.push_str(&text[cursor_pos + cursor.len()..]);
    let cursor_pos = TextUnit::from(cursor_pos as u32);
    (cursor_pos, new_text)
}

pub fn extract_range(text: &str) -> (TextRange, String) {
    let (start, text) = extract_offset(text);
    let (end, text) = extract_offset(&text);
    (TextRange::from_to(start, end), text)
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
