use std::fmt;

use camino::Utf8Path;

/// A location in a file.
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq, Eq, Debug))]
pub(crate) struct Location<'a> {
    pub(crate) file_path: &'a Utf8Path,
    pub(crate) line_number: LineNumber,
}

impl fmt::Display for Location<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file_path, self.line_number)
    }
}

/// A line number in a file.
///
/// The line number is basically one-based, meaning the first (real) line of a file is numbered `1`.
/// "Basically" one-based because there's still a [`Self::ZERO`] which represents "no specific line"
/// to be used for synthetic directives or in tests.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct LineNumber(usize);

impl LineNumber {
    /// This represents "no specific line" (used for synthetic directives and in tests).
    // FIXME: Get rid of this variant once synthetic directives are no more.
    pub(crate) const ZERO: Self = Self(0);

    /// A never ending iterator over line numbers starting from the first line (so `1`).
    pub(crate) fn enumerate() -> impl Iterator<Item = LineNumber> {
        (1..).map(LineNumber)
    }
}

impl fmt::Display for LineNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
