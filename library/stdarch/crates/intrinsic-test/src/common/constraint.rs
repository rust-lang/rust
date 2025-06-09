use serde::Deserialize;
use std::ops::Range;

#[derive(Debug, PartialEq, Clone, Deserialize)]
pub enum Constraint {
    Equal(i64),
    Range(Range<i64>),
}

impl Constraint {
    pub fn to_range(&self) -> Range<i64> {
        match self {
            Constraint::Equal(eq) => *eq..*eq + 1,
            Constraint::Range(range) => range.clone(),
        }
    }
}
