use std::fmt;

use itertools::Itertools;
use text_unit::{TextRange, TextUnit};
use serde_json::Value;

pub use difference::Changeset as __Changeset;

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

// Comparison functionality borrowed from cargo:

/// Compare a line with an expected pattern.
/// - Use `[..]` as a wildcard to match 0 or more characters on the same line
///   (similar to `.*` in a regex).
pub fn lines_match(expected: &str, actual: &str) -> bool {
    // Let's not deal with / vs \ (windows...)
    // First replace backslash-escaped backslashes with forward slashes
    // which can occur in, for example, JSON output
    let expected = expected.replace("\\\\", "/").replace("\\", "/");
    let mut actual: &str = &actual.replace("\\\\", "/").replace("\\", "/");
    for (i, part) in expected.split("[..]").enumerate() {
        match actual.find(part) {
            Some(j) => {
                if i == 0 && j != 0 {
                    return false;
                }
                actual = &actual[j + part.len()..];
            }
            None => return false,
        }
    }
    actual.is_empty() || expected.ends_with("[..]")
}

#[test]
fn lines_match_works() {
    assert!(lines_match("a b", "a b"));
    assert!(lines_match("a[..]b", "a b"));
    assert!(lines_match("a[..]", "a b"));
    assert!(lines_match("[..]", "a b"));
    assert!(lines_match("[..]b", "a b"));

    assert!(!lines_match("[..]b", "c"));
    assert!(!lines_match("b", "c"));
    assert!(!lines_match("b", "cb"));
}

// Compares JSON object for approximate equality.
// You can use `[..]` wildcard in strings (useful for OS dependent things such
// as paths).  You can use a `"{...}"` string literal as a wildcard for
// arbitrary nested JSON (useful for parts of object emitted by other programs
// (e.g. rustc) rather than Cargo itself).  Arrays are sorted before comparison.
pub fn find_mismatch<'a>(expected: &'a Value, actual: &'a Value) -> Option<(&'a Value, &'a Value)> {
    use serde_json::Value::*;
    match (expected, actual) {
        (&Number(ref l), &Number(ref r)) if l == r => None,
        (&Bool(l), &Bool(r)) if l == r => None,
        (&String(ref l), &String(ref r)) if lines_match(l, r) => None,
        (&Array(ref l), &Array(ref r)) => {
            if l.len() != r.len() {
                return Some((expected, actual));
            }

            let mut l = l.iter().collect::<Vec<_>>();
            let mut r = r.iter().collect::<Vec<_>>();

            l.retain(
                |l| match r.iter().position(|r| find_mismatch(l, r).is_none()) {
                    Some(i) => {
                        r.remove(i);
                        false
                    }
                    None => true,
                },
            );

            if !l.is_empty() {
                assert!(!r.is_empty());
                Some((&l[0], &r[0]))
            } else {
                assert_eq!(r.len(), 0);
                None
            }
        }
        (&Object(ref l), &Object(ref r)) => {
            let same_keys = l.len() == r.len() && l.keys().all(|k| r.contains_key(k));
            if !same_keys {
                return Some((expected, actual));
            }

            l.values()
                .zip(r.values())
                .filter_map(|(l, r)| find_mismatch(l, r))
                .nth(0)
        }
        (&Null, &Null) => None,
        // magic string literal "{...}" acts as wildcard for any sub-JSON
        (&String(ref l), _) if l == "{...}" => None,
        _ => Some((expected, actual)),
    }
}
