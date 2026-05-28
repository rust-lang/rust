use serde::Deserialize;
use std::ops::Range;

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

impl Constraint {
    /// Returns an iterator over the values of this constraint
    pub fn iter(&self) -> Box<dyn Iterator<Item = i64> + '_> {
        match self {
            Constraint::Equal(i) => Box::new(std::iter::once(*i)),
            Constraint::Range(range) => Box::new(range.clone()),
            Constraint::Set(items) => Box::new(items.iter().copied().chain(Range::default())),
            // These values are discriminants of the `svpattern` enum
            Constraint::SvPattern => Box::new((0..=13).chain(29..=31)),
            // These values are discriminants of the `svprfop` enum
            Constraint::SvPrefetchOp => Box::new((0..=5).chain(8..=14)),
            // Valid rotations for intrinsics operating on complex pairs: 0, 90, 180, 270
            Constraint::SvImmRotation => Box::new((0..=270).step_by(90)),
            // Valid rotations for `svcadd` and `svqcadd`: 0, 270
            Constraint::SvImmRotationAdd => Box::new((90..=270).step_by(180)),
        }
    }
}
