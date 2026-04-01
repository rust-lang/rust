use serde::Deserialize;
use std::ops::Range;

/// Describes the values to test for a const generic parameter.
#[derive(Debug, PartialEq, Clone, Deserialize)]
pub enum Constraint {
    /// Test a single value.
    Equal(i64),
    /// Test a range of values, e.g. `0..16`.
    Range(Range<i64>),
    /// Test discrete values, e.g. `vec![1, 2, 4, 8]`.
    Set(Vec<i64>),
}

impl Constraint {
    /// Iterate over the values of this constraint.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = i64> + 'a {
        match self {
            Constraint::Equal(i) => std::slice::Iter::default().copied().chain(*i..*i + 1),
            Constraint::Range(range) => std::slice::Iter::default().copied().chain(range.clone()),
            Constraint::Set(items) => items.iter().copied().chain(std::ops::Range::default()),
        }
    }
}
