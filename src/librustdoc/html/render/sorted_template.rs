use std::collections::BTreeSet;
use std::fmt::{self, Write as _};
use std::marker::PhantomData;
use std::str::FromStr;

use itertools::{Itertools as _, Position};
use serde::{Deserialize, Serialize};

/// Append-only templates for sorted, deduplicated lists of items.
///
/// Last line of the rendered output is a comment encoding the next insertion point.
#[derive(Debug, Clone)]
pub(crate) struct SortedTemplate<F> {
    format: PhantomData<F>,
    before: String,
    after: String,
    fragments: BTreeSet<String>,
}

/// Written to last line of file to specify the location of each fragment
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Offset {
    /// Index of the first byte in the template
    start: usize,
    /// The length of each fragment in the encoded template, including the separator
    fragment_lengths: Vec<usize>,
}

impl<F> SortedTemplate<F> {
    /// Generate this template from arbitrary text.
    /// Will insert wherever the substring `delimiter` can be found.
    /// Errors if it does not appear exactly once.
    pub(crate) fn from_template(template: &str, delimiter: &str) -> Result<Self, Error> {
        let mut split = template.split(delimiter);
        let before = split.next().ok_or(Error("delimiter should appear at least once"))?;
        let after = split.next().ok_or(Error("delimiter should appear at least once"))?;
        // not `split_once` because we want to check for too many occurrences
        if split.next().is_some() {
            return Err(Error("delimiter should appear at most once"));
        }
        Ok(Self::from_before_after(before, after))
    }

    /// Template will insert fragments between `before` and `after`
    pub(crate) fn from_before_after<S: ToString, T: ToString>(before: S, after: T) -> Self {
        let before = before.to_string();
        let after = after.to_string();
        Self { format: PhantomData, before, after, fragments: Default::default() }
    }
}

impl<F> SortedTemplate<F> {
    /// Adds this text to the template
    pub(crate) fn append(&mut self, insert: String) {
        self.fragments.insert(insert);
    }
}

impl<F: FileFormat> fmt::Display for SortedTemplate<F> {
    fn fmt(&self, mut f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut fragment_lengths = Vec::default();
        write!(f, "{}", self.before)?;
        for (p, fragment) in self.fragments.iter().with_position() {
            let mut f = DeltaWriter { inner: &mut f, delta: 0 };
            let sep = if matches!(p, Position::First | Position::Only) { "" } else { F::SEPARATOR };
            f.write_str(sep)?;
            f.write_str(fragment)?;
            fragment_lengths.push(f.delta);
        }
        let offset = Offset { start: self.before.len(), fragment_lengths };
        let offset = serde_json::to_string(&offset).unwrap();
        write!(f, "{}\n{}{}{}", self.after, F::COMMENT_START, offset, F::COMMENT_END)
    }
}

impl<F: FileFormat> FromStr for SortedTemplate<F> {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (s, offset) = s
            .rsplit_once("\n")
            .ok_or(Error("invalid format: should have a newline on the last line"))?;
        let offset = offset
            .strip_prefix(F::COMMENT_START)
            .ok_or(Error("last line expected to start with a comment"))?;
        let offset = offset
            .strip_suffix(F::COMMENT_END)
            .ok_or(Error("last line expected to end with a comment"))?;
        let offset: Offset = serde_json::from_str(offset).map_err(|_| {
            Error("could not find insertion location descriptor object on last line")
        })?;
        let (before, mut s) =
            s.split_at_checked(offset.start).ok_or(Error("invalid start: out of bounds"))?;
        let mut fragments = BTreeSet::default();
        for (p, &index) in offset.fragment_lengths.iter().with_position() {
            let (fragment, rest) =
                s.split_at_checked(index).ok_or(Error("invalid fragment length: out of bounds"))?;
            s = rest;
            let sep = if matches!(p, Position::First | Position::Only) { "" } else { F::SEPARATOR };
            let fragment = fragment
                .strip_prefix(sep)
                .ok_or(Error("invalid fragment length: expected to find separator here"))?;
            fragments.insert(fragment.to_string());
        }
        Ok(Self {
            format: PhantomData,
            before: before.to_string(),
            after: s.to_string(),
            fragments,
        })
    }
}

pub(crate) trait FileFormat {
    const COMMENT_START: &'static str;
    const COMMENT_END: &'static str;
    const SEPARATOR: &'static str;
}

#[derive(Debug, Clone)]
pub(crate) struct Html;

impl FileFormat for Html {
    const COMMENT_START: &'static str = "<!--";
    const COMMENT_END: &'static str = "-->";
    const SEPARATOR: &'static str = "";
}

#[derive(Debug, Clone)]
pub(crate) struct Js;

impl FileFormat for Js {
    const COMMENT_START: &'static str = "//";
    const COMMENT_END: &'static str = "";
    const SEPARATOR: &'static str = ",";
}

#[derive(Debug, Clone)]
pub(crate) struct Error(&'static str);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid template: {}", self.0)
    }
}

struct DeltaWriter<W> {
    inner: W,
    delta: usize,
}

impl<W: fmt::Write> fmt::Write for DeltaWriter<W> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.inner.write_str(s)?;
        self.delta += s.len();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
