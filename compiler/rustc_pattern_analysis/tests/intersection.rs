//! Test the computation of arm intersections.

use common::*;
use rustc_pattern_analysis::{pat::DeconstructedPat, usefulness::PlaceValidity, MatchArm};

#[macro_use]
mod common;

/// Analyze a match made of these patterns and returns the computed arm intersections.
fn check(patterns: Vec<DeconstructedPat<Cx>>) -> Vec<Vec<usize>> {
    let ty = *patterns[0].ty();
    let arms: Vec<_> =
        patterns.iter().map(|pat| MatchArm { pat, has_guard: false, arm_data: () }).collect();
    let report =
        compute_match_usefulness(arms.as_slice(), ty, PlaceValidity::ValidOnly, None).unwrap();
    report.arm_intersections.into_iter().map(|bitset| bitset.iter().collect()).collect()
}

#[track_caller]
fn assert_intersects(patterns: Vec<DeconstructedPat<Cx>>, intersects: &[&[usize]]) {
    let computed_intersects = check(patterns);
    assert_eq!(computed_intersects, intersects);
}

#[test]
fn test_int_ranges() {
    let ty = Ty::U8;
    assert_intersects(
        pats!(ty;
            0..=100,
            100..,
        ),
        &[&[], &[0]],
    );
    assert_intersects(
        pats!(ty;
            0..=101,
            100..,
        ),
        &[&[], &[0]],
    );
    assert_intersects(
        pats!(ty;
            0..100,
            100..,
        ),
        &[&[], &[]],
    );
}

#[test]
fn test_nested() {
    let ty = Ty::Tuple(&[Ty::Bool; 2]);
    assert_intersects(
        pats!(ty;
            (true, true),
            (true, _),
            (_, true),
        ),
        &[&[], &[0], &[0, 1]],
    );
    // Here we shortcut because `(true, true)` is irrelevant, so we fail to detect the intersection.
    assert_intersects(
        pats!(ty;
            (true, _),
            (_, true),
        ),
        &[&[], &[]],
    );
}
