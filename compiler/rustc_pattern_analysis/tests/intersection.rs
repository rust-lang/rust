//! Test the computation of arm intersections.

#![allow(unused_crate_dependencies)]

use common::*;
use rustc_pattern_analysis::MatchArm;
use rustc_pattern_analysis::pat::DeconstructedPat;
use rustc_pattern_analysis::usefulness::PlaceValidity;

#[macro_use]
mod common;

/// Analyze a match made of these patterns and returns the computed arm intersections.
fn check(patterns: Vec<DeconstructedPat<Cx>>) -> Vec<Vec<usize>> {
    let ty = *patterns[0].ty();
    let arms: Vec<_> =
        patterns.iter().map(|pat| MatchArm { pat, has_guard: false, arm_data: () }).collect();
    let report =
        compute_match_usefulness(arms.as_slice(), ty, PlaceValidity::ValidOnly, usize::MAX, false)
            .unwrap();
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
    let ty = Ty::Tuple(&[Ty::Bool; 3]);
    assert_intersects(
        pats!(ty;
            (true, true, _),
            (true, _, true),
            (false, _, _),
        ),
        &[&[], &[], &[]],
    );
    let ty = Ty::Tuple(&[Ty::Bool, Ty::Bool, Ty::U8]);
    assert_intersects(
        pats!(ty;
            (true, _, _),
            (_, true, 0..10),
            (_, true, 10..),
            (_, true, 3),
            _,
        ),
        &[&[], &[], &[], &[1], &[0, 1, 2, 3]],
    );
}
