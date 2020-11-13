use crate::fmt::Debug;

/// Information about a failed assertion.
#[derive(Debug)]
pub struct AssertInfo<'a> {
    /// The assertion that failed.
    pub assertion: Assertion<'a>,
    /// Optional additional message to include in the failure report.
    pub message: Option<crate::fmt::Arguments<'a>>,
}

/// Details about the expression that failed an assertion.
#[derive(Debug)]
pub enum Assertion<'a> {
    /// The failed assertion is a boolean expression.
    ///
    /// This variant is only used for expressions that can't be described more specifically
    /// by another variant.
    Bool(BoolAssertion),

    /// The failed assertion is a binary comparison expression.
    ///
    /// This is used by `assert_eq!()`, `assert_ne!()` and expressions like
    /// `assert!(x > 10)`.
    Binary(BinaryAssertion<'a>),
}

/// Information about a failed boolean assertion.
///
/// The expression was asserted to be true, but it evaluated to false.
///
/// This struct is only used for assertion failures that can't be described more specifically
/// by another assertion type.
#[derive(Debug)]
pub struct BoolAssertion {
    /// The expression that was evaluated to false.
    pub expr: &'static str,
}

/// Information about a failed binary comparison assertion.
///
/// The left expression was compared with the right expression using `op`,
/// and the comparison evaluted to false.
///
/// This struct is used for `assert_eq!()`, `assert_ne!()` and expressions like
/// `assert!(x > 10)`.
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
