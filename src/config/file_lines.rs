// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains types and functions to support formatting specific line ranges.

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::{cmp, fmt, iter, str};

use serde::de::{Deserialize, Deserializer};
use serde::ser::{self, Serialize, Serializer};
use serde_json as json;

use syntax::source_map::{self, SourceFile};

/// A range of lines in a file, inclusive of both ends.
pub struct LineRange {
    pub file: Rc<SourceFile>,
    pub lo: usize,
    pub hi: usize,
}

/// Defines the name of an input - either a file or stdin.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum FileName {
    Real(PathBuf),
    Stdin,
}

impl From<source_map::FileName> for FileName {
    fn from(name: source_map::FileName) -> FileName {
        match name {
            source_map::FileName::Real(p) => FileName::Real(p),
            source_map::FileName::Custom(ref f) if f == "stdin" => FileName::Stdin,
            _ => unreachable!(),
        }
    }
}

impl fmt::Display for FileName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FileName::Real(p) => write!(f, "{}", p.to_str().unwrap()),
            FileName::Stdin => write!(f, "stdin"),
        }
    }
}

impl<'de> Deserialize<'de> for FileName {
    fn deserialize<D>(deserializer: D) -> Result<FileName, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s == "stdin" {
            Ok(FileName::Stdin)
        } else {
            Ok(FileName::Real(s.into()))
        }
    }
}

impl Serialize for FileName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = match self {
            FileName::Stdin => Ok("stdin"),
            FileName::Real(path) => path
                .to_str()
                .ok_or_else(|| ser::Error::custom("path can't be serialized as UTF-8 string")),
        };

        s.and_then(|s| serializer.serialize_str(s))
    }
}

impl LineRange {
    pub fn file_name(&self) -> FileName {
        self.file.name.clone().into()
    }
}

/// A range that is inclusive of both ends.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Deserialize)]
pub struct Range {
    lo: usize,
    hi: usize,
}

impl<'a> From<&'a LineRange> for Range {
    fn from(range: &'a LineRange) -> Range {
        Range::new(range.lo, range.hi)
    }
}

impl Range {
    pub fn new(lo: usize, hi: usize) -> Range {
        Range { lo, hi }
    }

    fn is_empty(self) -> bool {
        self.lo > self.hi
    }

    #[allow(dead_code)]
    fn contains(self, other: Range) -> bool {
        if other.is_empty() {
            true
        } else {
            !self.is_empty() && self.lo <= other.lo && self.hi >= other.hi
        }
    }

    fn intersects(self, other: Range) -> bool {
        if self.is_empty() || other.is_empty() {
            false
        } else {
            (self.lo <= other.hi && other.hi <= self.hi)
                || (other.lo <= self.hi && self.hi <= other.hi)
        }
    }

    fn adjacent_to(self, other: Range) -> bool {
        if self.is_empty() || other.is_empty() {
            false
        } else {
            self.hi + 1 == other.lo || other.hi + 1 == self.lo
        }
    }

    /// Returns a new `Range` with lines from `self` and `other` if they were adjacent or
    /// intersect; returns `None` otherwise.
    fn merge(self, other: Range) -> Option<Range> {
        if self.adjacent_to(other) || self.intersects(other) {
            Some(Range::new(
                cmp::min(self.lo, other.lo),
                cmp::max(self.hi, other.hi),
            ))
        } else {
            None
        }
    }
}

/// A set of lines in files.
///
/// It is represented as a multimap keyed on file names, with values a collection of
/// non-overlapping ranges sorted by their start point. An inner `None` is interpreted to mean all
/// lines in all files.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FileLines(Option<HashMap<FileName, Vec<Range>>>);

/// Normalizes the ranges so that the invariants for `FileLines` hold: ranges are non-overlapping,
/// and ordered by their start point.
fn normalize_ranges(ranges: &mut HashMap<FileName, Vec<Range>>) {
    for ranges in ranges.values_mut() {
        ranges.sort();
        let mut result = vec![];
        {
            let mut iter = ranges.into_iter().peekable();
            while let Some(next) = iter.next() {
                let mut next = *next;
                while let Some(&&mut peek) = iter.peek() {
                    if let Some(merged) = next.merge(peek) {
                        iter.next().unwrap();
                        next = merged;
                    } else {
                        break;
                    }
                }
                result.push(next)
            }
        }
        *ranges = result;
    }
}

impl FileLines {
    /// Creates a `FileLines` that contains all lines in all files.
    pub(crate) fn all() -> FileLines {
        FileLines(None)
    }

    /// Returns true if this `FileLines` contains all lines in all files.
    pub(crate) fn is_all(&self) -> bool {
        self.0.is_none()
    }

    pub fn from_ranges(mut ranges: HashMap<FileName, Vec<Range>>) -> FileLines {
        normalize_ranges(&mut ranges);
        FileLines(Some(ranges))
    }

    /// Returns an iterator over the files contained in `self`.
    pub fn files(&self) -> Files {
        Files(self.0.as_ref().map(|m| m.keys()))
    }

    /// Returns JSON representation as accepted by the `--file-lines JSON` arg.
    pub fn to_json_spans(&self) -> Vec<JsonSpan> {
        match &self.0 {
            None => vec![],
            Some(file_ranges) => file_ranges
                .iter()
                .flat_map(|(file, ranges)| ranges.iter().map(move |r| (file, r)))
                .map(|(file, range)| JsonSpan {
                    file: file.to_owned(),
                    range: (range.lo, range.hi),
                }).collect(),
        }
    }

    /// Returns true if `self` includes all lines in all files. Otherwise runs `f` on all ranges in
    /// the designated file (if any) and returns true if `f` ever does.
    fn file_range_matches<F>(&self, file_name: &FileName, f: F) -> bool
    where
        F: FnMut(&Range) -> bool,
    {
        let map = match self.0 {
            // `None` means "all lines in all files".
            None => return true,
            Some(ref map) => map,
        };

        match canonicalize_path_string(file_name).and_then(|file| map.get(&file)) {
            Some(ranges) => ranges.iter().any(f),
            None => false,
        }
    }

    /// Returns true if `range` is fully contained in `self`.
    #[allow(dead_code)]
    pub(crate) fn contains(&self, range: &LineRange) -> bool {
        self.file_range_matches(&range.file_name(), |r| r.contains(Range::from(range)))
    }

    /// Returns true if any lines in `range` are in `self`.
    pub(crate) fn intersects(&self, range: &LineRange) -> bool {
        self.file_range_matches(&range.file_name(), |r| r.intersects(Range::from(range)))
    }

    /// Returns true if `line` from `file_name` is in `self`.
    pub(crate) fn contains_line(&self, file_name: &FileName, line: usize) -> bool {
        self.file_range_matches(file_name, |r| r.lo <= line && r.hi >= line)
    }

    /// Returns true if all the lines between `lo` and `hi` from `file_name` are in `self`.
    pub(crate) fn contains_range(&self, file_name: &FileName, lo: usize, hi: usize) -> bool {
        self.file_range_matches(file_name, |r| r.contains(Range::new(lo, hi)))
    }
}

/// `FileLines` files iterator.
pub struct Files<'a>(Option<::std::collections::hash_map::Keys<'a, FileName, Vec<Range>>>);

impl<'a> iter::Iterator for Files<'a> {
    type Item = &'a FileName;

    fn next(&mut self) -> Option<&'a FileName> {
        self.0.as_mut().and_then(Iterator::next)
    }
}

fn canonicalize_path_string(file: &FileName) -> Option<FileName> {
    match *file {
        FileName::Real(ref path) => path.canonicalize().ok().map(FileName::Real),
        _ => Some(file.clone()),
    }
}

// This impl is needed for `Config::override_value` to work for use in tests.
impl str::FromStr for FileLines {
    type Err = String;

    fn from_str(s: &str) -> Result<FileLines, String> {
        let v: Vec<JsonSpan> = json::from_str(s).map_err(|e| e.to_string())?;
        let mut m = HashMap::new();
        for js in v {
            let (s, r) = JsonSpan::into_tuple(js)?;
            m.entry(s).or_insert_with(|| vec![]).push(r);
        }
        Ok(FileLines::from_ranges(m))
    }
}

// For JSON decoding.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Deserialize, Serialize)]
pub struct JsonSpan {
    file: FileName,
    range: (usize, usize),
}

impl JsonSpan {
    fn into_tuple(self) -> Result<(FileName, Range), String> {
        let (lo, hi) = self.range;
        let canonical = canonicalize_path_string(&self.file)
            .ok_or_else(|| format!("Can't canonicalize {}", &self.file))?;
        Ok((canonical, Range::new(lo, hi)))
    }
}

// This impl is needed for inclusion in the `Config` struct. We don't have a toml representation
// for `FileLines`, so it will just panic instead.
impl<'de> ::serde::de::Deserialize<'de> for FileLines {
    fn deserialize<D>(_: D) -> Result<Self, D::Error>
    where
        D: ::serde::de::Deserializer<'de>,
    {
        panic!(
            "FileLines cannot be deserialized from a project rustfmt.toml file: please \
             specify it via the `--file-lines` option instead"
        );
    }
}

// We also want to avoid attempting to serialize a FileLines to toml. The
// `Config` struct should ensure this impl is never reached.
impl ::serde::ser::Serialize for FileLines {
    fn serialize<S>(&self, _: S) -> Result<S::Ok, S::Error>
    where
        S: ::serde::ser::Serializer,
    {
        unreachable!("FileLines cannot be serialized. This is a rustfmt bug.");
    }
}

#[cfg(test)]
mod test {
    use super::Range;

    #[test]
    fn test_range_intersects() {
        assert!(Range::new(1, 2).intersects(Range::new(1, 1)));
        assert!(Range::new(1, 2).intersects(Range::new(2, 2)));
        assert!(!Range::new(1, 2).intersects(Range::new(0, 0)));
        assert!(!Range::new(1, 2).intersects(Range::new(3, 10)));
        assert!(!Range::new(1, 3).intersects(Range::new(5, 5)));
    }

    #[test]
    fn test_range_adjacent_to() {
        assert!(!Range::new(1, 2).adjacent_to(Range::new(1, 1)));
        assert!(!Range::new(1, 2).adjacent_to(Range::new(2, 2)));
        assert!(Range::new(1, 2).adjacent_to(Range::new(0, 0)));
        assert!(Range::new(1, 2).adjacent_to(Range::new(3, 10)));
        assert!(!Range::new(1, 3).adjacent_to(Range::new(5, 5)));
    }

    #[test]
    fn test_range_contains() {
        assert!(Range::new(1, 2).contains(Range::new(1, 1)));
        assert!(Range::new(1, 2).contains(Range::new(2, 2)));
        assert!(!Range::new(1, 2).contains(Range::new(0, 0)));
        assert!(!Range::new(1, 2).contains(Range::new(3, 10)));
    }

    #[test]
    fn test_range_merge() {
        assert_eq!(None, Range::new(1, 3).merge(Range::new(5, 5)));
        assert_eq!(None, Range::new(4, 7).merge(Range::new(0, 1)));
        assert_eq!(
            Some(Range::new(3, 7)),
            Range::new(3, 5).merge(Range::new(4, 7))
        );
        assert_eq!(
            Some(Range::new(3, 7)),
            Range::new(3, 5).merge(Range::new(5, 7))
        );
        assert_eq!(
            Some(Range::new(3, 7)),
            Range::new(3, 5).merge(Range::new(6, 7))
        );
        assert_eq!(
            Some(Range::new(3, 7)),
            Range::new(3, 7).merge(Range::new(4, 5))
        );
    }

    use super::json::{self, json};
    use super::{FileLines, FileName};
    use std::{collections::HashMap, path::PathBuf};

    #[test]
    fn file_lines_to_json() {
        let ranges: HashMap<FileName, Vec<Range>> = [
            (
                FileName::Real(PathBuf::from("src/main.rs")),
                vec![Range::new(1, 3), Range::new(5, 7)],
            ),
            (
                FileName::Real(PathBuf::from("src/lib.rs")),
                vec![Range::new(1, 7)],
            ),
        ]
            .iter()
            .cloned()
            .collect();

        let file_lines = FileLines::from_ranges(ranges);
        let mut spans = file_lines.to_json_spans();
        spans.sort();
        let json = json::to_value(&spans).unwrap();
        assert_eq!(
            json,
            json! {[
                {"file": "src/lib.rs",  "range": [1, 7]},
                {"file": "src/main.rs", "range": [1, 3]},
                {"file": "src/main.rs", "range": [5, 7]},
            ]}
        );
    }
}
