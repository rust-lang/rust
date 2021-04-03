use crate::fmt::Debug;

/// Information about a failed assertion.
#[derive(Debug)]
pub struct AssertInfo<'a> {
    /// The assertion that failed.
    pub assertion: Assertion<'a>,
    /// Optional additional message to include in the failure report.
    pub message: Option<crate::fmt::Arguments<'a>>,
}

#[derive(Debug)]
pub enum Assertion<'a> {
    /// The assertion is a boolean assertion.
    Bool(BoolAssertion),
    /// The assertion is a binary comparison assertion.
    Binary(BinaryAssertion<'a>),
}

/// Information about a failed boolean assertion.
#[derive(Debug)]
pub struct BoolAssertion {
    /// The expression that was evaluated.
    pub expr: &'static str,
}

/// Information about a failed binary comparison assertion.
#[derive(Debug)]
pub struct BinaryAssertion<'a> {
    /// The operator used to compare left and right.
    pub op: &'static str,
    /// The left expression as string.
    pub left_expr: &'static str,
    /// The right expression as string.
    pub right_expr: &'static str,
    /// The value of the left expression.
    pub left_val: &'a dyn Debug,
    /// The value of the right expression.
    pub right_val: &'a dyn Debug,
}

impl<'a> From<BoolAssertion> for Assertion<'a> {
    fn from(other: BoolAssertion) -> Self {
        Self::Bool(other)
    }
}

impl<'a> From<BinaryAssertion<'a>> for Assertion<'a> {
    fn from(other: BinaryAssertion<'a>) -> Self {
        Self::Binary(other)
    }
}
