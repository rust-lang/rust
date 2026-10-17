/// A line number in a file. Internally the first line has index 1.
/// If it is 0 it means "no specific line" (used e.g. for implied directives).
/// When `Display`:ed, the first line is `1`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct LineNumber(usize);

impl LineNumber {
    /// This represents "no specific line" (used e.g. for implied directives).
    pub(crate) const ZERO: Self = Self(0);

    /// A never ending iterator over line numbers starting from the first line.
    pub(crate) fn enumerate() -> impl Iterator<Item = LineNumber> {
        (1..).map(LineNumber)
    }
}

impl std::fmt::Display for LineNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
