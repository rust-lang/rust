use std::ops::Range;

use serde::Deserialize;

/// Describes the values to test for a const generic parameter
#[derive(Debug, PartialEq, Clone, Deserialize)]
pub enum Constraint {
    /// Test a single value
    Equal(i64),
    /// Test a range of values, e.g. `0..16`.
    Range(Range<i64>),
    /// Test discrete values, e.g. `vec![1, 2, 4, 8]`.
    Set(Vec<i64>),
    /// Values of `core::arch::aarch64::svpattern`
    SvPattern,
    /// Values of `core::arch::aarch64::svprfop`
    SvPrefetchOp,
    // Values of the `imm_rotation` argument in SVE intrinsics where arguments contain complex
    // pairs and `imm_rotation` corresponds to the rotation.
    SvImmRotation,
    // Values of the `imm_rotation` argument in SVE intrinsics where arguments contain complex
    // pairs and `imm_rotation` corresponds to the rotation (this variant is specifically for
    // `svcadd` and `svqcadd` where only 90 and 270 are valid arguments).
    SvImmRotationAdd,
}

/// Workaround to enable the `Constraint::into_iter` to return an iterator that implements `Clone`,
/// so that it can be used with `Itertools::multi_cartesian_product`.
///
/// With the different iterator types, returning `Box<dyn Iterator<Item = i64> + '_>` would the
/// idiomatic approach, but this can't be made to implement `Clone`. Given the limited number
/// of iterator types used and their relative lack of complexity, wrapping them all in an enum isn't
/// too bad.
#[derive(Clone)]
pub enum ConstraintIterator<'a> {
    Once(std::iter::Once<i64>),
    Range(std::ops::Range<i64>),
    Copied(std::iter::Copied<std::slice::Iter<'a, i64>>),
    Chain(std::iter::Chain<std::ops::RangeInclusive<i64>, std::ops::RangeInclusive<i64>>),
    StepBy(std::iter::StepBy<std::ops::RangeInclusive<i64>>),
}

impl<'a> Iterator for ConstraintIterator<'a> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ConstraintIterator::Once(once) => once.next(),
            ConstraintIterator::Range(range) => range.next(),
            ConstraintIterator::Copied(copied) => copied.next(),
            ConstraintIterator::Chain(chain) => chain.next(),
            ConstraintIterator::StepBy(step_by) => step_by.next(),
        }
    }
}

impl<'a> IntoIterator for &'a Constraint {
    type Item = i64;
    type IntoIter = ConstraintIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Constraint::Equal(i) => ConstraintIterator::Once(std::iter::once(*i)),
            Constraint::Range(range) => ConstraintIterator::Range(range.clone()),
            Constraint::Set(items) => ConstraintIterator::Copied(items.iter().copied()),
            // These values are discriminants of the `svpattern` enum
            Constraint::SvPattern => ConstraintIterator::Chain((0..=13).chain(29..=31)),
            // These values are discriminants of the `svprfop` enum
            Constraint::SvPrefetchOp => ConstraintIterator::Chain((0..=5).chain(8..=14)),
            // Valid rotations for intrinsics operating on complex pairs: 0, 90, 180, 270
            Constraint::SvImmRotation => ConstraintIterator::StepBy((0..=270).step_by(90)),
            // Valid rotations for `svcadd` and `svqcadd`: 0, 270
            Constraint::SvImmRotationAdd => ConstraintIterator::StepBy((90..=270).step_by(180)),
        }
    }
}
