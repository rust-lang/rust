use crate::fmt::{self, Debug};

/// Information about a failed assertion.
#[derive(Debug)]
pub struct AssertInfo<'a> {
    /// The assertion that failed.
    pub assertion: Assertion<'a>,
    /// Optional additional message to include in the failure report.
    pub message: Option<crate::fmt::Arguments<'a>>,
}

impl fmt::Display for AssertInfo<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.assertion {
            Assertion::Binary(ref assertion) => match self.message {
                Some(message) => write!(
                    formatter,
                    r#"assertion failed: `(left {} right)`
  left: `{:?}`,
 right: `{:?}`: {}"#,
                    assertion.static_data.kind.op(),
                    assertion.left_val,
                    assertion.right_val,
                    message
                ),
                None => write!(
                    formatter,
                    r#"assertion failed: `(left {} right)`
  left: `{:?}`,
 right: `{:?}`"#,
                    assertion.static_data.kind.op(),
                    assertion.left_val,
                    assertion.right_val
                ),
            },
        }
    }
}

/// Details about the expression that failed an assertion.
#[derive(Debug)]
pub enum Assertion<'a> {
    /// The failed assertion is a binary expression.
    Binary(BinaryAssertion<'a>),
}

/// Information about a failed binary assertion.
#[derive(Debug)]
pub struct BinaryAssertion<'a> {
    /// Static information about the failed assertion.
    pub static_data: &'static BinaryAssertionStaticData,
    /// The left value of the binary assertion.
    pub left_val: &'a dyn Debug,
    /// The right value of the binary assertion.
    pub right_val: &'a dyn Debug,
}

/// Information about a binary assertion that can be constructed at compile time.
///
/// This struct helps to reduce the size `AssertInfo`.
#[derive(Debug)]
pub struct BinaryAssertionStaticData {
    /// The kind of the binary assertion
    pub kind: BinaryAssertKind,
    /// The left expression of the binary assertion.
    pub left_expr: &'static str,
    /// The right expression of the binary assertion.
    pub right_expr: &'static str,
}

/// The kind of a binary assertion
#[derive(Debug)]
pub enum BinaryAssertKind {
    Eq,
    Ne,
    Match,
}

impl BinaryAssertKind {
    /// The name of the macro that triggered the panic.
    pub fn macro_name(&self) -> &'static str {
        match self {
            Self::Eq { .. } => "assert_eq",
            Self::Ne { .. } => "assert_ne",
            Self::Match { .. } => "assert_matches",
        }
    }

    /// Symbolic representation of the binary assertion.
    pub fn op(&self) -> &'static str {
        match self {
            Self::Eq { .. } => "==",
            Self::Ne { .. } => "!=",
            Self::Match { .. } => "matches",
        }
    }
}
