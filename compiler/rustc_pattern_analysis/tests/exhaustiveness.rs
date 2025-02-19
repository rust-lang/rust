//! Test exhaustiveness checking.

use common::*;
use rustc_pattern_analysis::MatchArm;
use rustc_pattern_analysis::pat::{DeconstructedPat, WitnessPat};
use rustc_pattern_analysis::usefulness::PlaceValidity;

#[macro_use]
mod common;

/// Analyze a match made of these patterns.
fn check(patterns: Vec<DeconstructedPat<Cx>>) -> Vec<WitnessPat<Cx>> {
    let ty = *patterns[0].ty();
    let arms: Vec<_> =
        patterns.iter().map(|pat| MatchArm { pat, has_guard: false, arm_data: () }).collect();
    let report =
        compute_match_usefulness(arms.as_slice(), ty, PlaceValidity::ValidOnly, usize::MAX)
            .unwrap();
    report.non_exhaustiveness_witnesses
}

#[track_caller]
fn assert_exhaustive(patterns: Vec<DeconstructedPat<Cx>>) {
    let witnesses = check(patterns);
    if !witnesses.is_empty() {
        panic!("non-exhaustive match: missing {witnesses:?}");
    }
}

#[track_caller]
fn assert_non_exhaustive(patterns: Vec<DeconstructedPat<Cx>>) {
    let witnesses = check(patterns);
    assert!(!witnesses.is_empty())
}

#[test]
fn test_int_ranges() {
    let ty = Ty::U8;
    assert_exhaustive(pats!(ty;
        0..=255,
    ));
    assert_exhaustive(pats!(ty;
        0..,
    ));
    assert_non_exhaustive(pats!(ty;
        0..255,
    ));
    assert_exhaustive(pats!(ty;
        0..255,
        255,
    ));
    assert_exhaustive(pats!(ty;
        ..10,
        10..
    ));
}

#[test]
fn test_nested() {
    let ty = Ty::BigStruct { arity: 2, ty: &Ty::BigEnum { arity: 2, ty: &Ty::Bool } };
    assert_non_exhaustive(pats!(ty;
        Struct(Variant.0, _),
    ));
    assert_exhaustive(pats!(ty;
        Struct(Variant.0, _),
        Struct(Variant.1, _),
    ));
    assert_non_exhaustive(pats!(ty;
        Struct(Variant.0, _),
        Struct(_, Variant.0),
    ));
    assert_exhaustive(pats!(ty;
        Struct(Variant.0, _),
        Struct(_, Variant.0),
        Struct(Variant.1, Variant.1),
    ));
}

#[test]
fn test_empty() {
    // `TY = Result<bool, !>`
    const TY: Ty = Ty::Enum(&[Ty::Bool, Ty::Enum(&[])]);
    assert_exhaustive(pats!(TY;
        Variant.0,
    ));
    let ty = Ty::Tuple(&[Ty::Bool, TY]);
    assert_exhaustive(pats!(ty;
        (true, Variant.0),
        (false, Variant.0),
    ));
}
