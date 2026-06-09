//! Test the pattern complexity limit.

#![allow(unused_crate_dependencies)]

use common::*;
use rustc_pattern_analysis::MatchArm;
use rustc_pattern_analysis::pat::DeconstructedPat;
use rustc_pattern_analysis::usefulness::PlaceValidity;

#[macro_use]
mod common;

/// Analyze a match made of these patterns. Ignore the report; we only care whether we exceeded the
/// limit or not.
fn check(patterns: &[DeconstructedPat<Cx>], complexity_limit: usize) -> Result<(), ()> {
    let ty = *patterns[0].ty();
    let arms: Vec<_> =
        patterns.iter().map(|pat| MatchArm { pat, has_guard: false, arm_data: () }).collect();
    compute_match_usefulness(arms.as_slice(), ty, PlaceValidity::ValidOnly, complexity_limit, false)
        .map(|_report| ())
}

/// Asserts that analyzing this match takes exactly `complexity` steps.
#[track_caller]
fn assert_complexity(patterns: Vec<DeconstructedPat<Cx>>, complexity: usize) {
    assert!(check(&patterns, complexity).is_ok());
    assert!(check(&patterns, complexity - 1).is_err());
}

/// Construct a match like:
/// ```ignore(illustrative)
/// match ... {
///     BigStruct { field01: true, .. } => {}
///     BigStruct { field02: true, .. } => {}
///     BigStruct { field03: true, .. } => {}
///     BigStruct { field04: true, .. } => {}
///     ...
///     _ => {}
/// }
/// ```
fn diagonal_match(arity: usize) -> Vec<DeconstructedPat<Cx>> {
    let struct_ty = Ty::BigStruct { arity, ty: &Ty::Bool };
    let mut patterns = vec![];
    for i in 0..arity {
        patterns.push(pat!(struct_ty; Struct { .i: true }));
    }
    patterns.push(pat!(struct_ty; _));
    patterns
}

/// Construct a match like:
/// ```ignore(illustrative)
/// match ... {
///     BigStruct { field01: true, .. } => {}
///     BigStruct { field02: true, .. } => {}
///     BigStruct { field03: true, .. } => {}
///     BigStruct { field04: true, .. } => {}
///     ...
///     BigStruct { field01: false, .. } => {}
///     BigStruct { field02: false, .. } => {}
///     BigStruct { field03: false, .. } => {}
///     BigStruct { field04: false, .. } => {}
///     ...
///     _ => {}
/// }
/// ```
fn diagonal_exponential_match(arity: usize) -> Vec<DeconstructedPat<Cx>> {
    let struct_ty = Ty::BigStruct { arity, ty: &Ty::Bool };
    let mut patterns = vec![];
    for i in 0..arity {
        patterns.push(pat!(struct_ty; Struct { .i: true }));
    }
    for i in 0..arity {
        patterns.push(pat!(struct_ty; Struct { .i: false }));
    }
    patterns.push(pat!(struct_ty; _));
    patterns
}

#[test]
fn test_diagonal_struct_match() {
    // These cases are nicely linear: we check `arity` patterns with exactly one `true`, matching
    // in 2 branches each, and a final pattern with all `false`, matching only the `_` branch.
    assert_complexity(diagonal_match(20), 41);
    assert_complexity(diagonal_match(30), 61);
    // This case goes exponential.
    assert!(check(&diagonal_exponential_match(10), 10000).is_err());
}

/// Construct a match like:
/// ```ignore(illustrative)
/// match ... {
///     BigEnum::Variant1(_) => {}
///     BigEnum::Variant2(_) => {}
///     BigEnum::Variant3(_) => {}
///     ...
///     _ => {}
/// }
/// ```
fn big_enum(arity: usize) -> Vec<DeconstructedPat<Cx>> {
    let enum_ty = Ty::BigEnum { arity, ty: &Ty::Bool };
    let mut patterns = vec![];
    for i in 0..arity {
        patterns.push(pat!(enum_ty; Variant.i));
    }
    patterns.push(pat!(enum_ty; _));
    patterns
}

#[test]
fn test_big_enum() {
    // We try 2 branches per variant.
    assert_complexity(big_enum(20), 40);
}
