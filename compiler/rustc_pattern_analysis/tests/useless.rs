//! Test detection of useless or-pattern alternatives, like `0` in `0 | _`.

#![allow(unused_crate_dependencies)]

use common::*;
use rustc_pattern_analysis::MatchArm;
use rustc_pattern_analysis::constructor::Constructor;
use rustc_pattern_analysis::pat::DeconstructedPat;
use rustc_pattern_analysis::usefulness::{PlaceValidity, Usefulness};

#[macro_use]
mod common;

/// Construct an or-pattern with the given alternatives.
fn or_pat(ty: Ty, alts: Vec<DeconstructedPat<Cx>>) -> DeconstructedPat<Cx> {
    let arity = alts.len();
    let fields = alts.into_iter().enumerate().map(|(i, pat)| pat.at_index(i)).collect();
    DeconstructedPat::new(Constructor::Or, fields, arity, ty, ())
}

/// Return the path of field indices leading to `target` in the subpattern tree of `root`.
fn find_path(root: &DeconstructedPat<Cx>, target: &DeconstructedPat<Cx>) -> Option<Vec<usize>> {
    if root == target {
        return Some(Vec::new());
    }
    for ipat in root.iter_fields() {
        if let Some(mut path) = find_path(&ipat.pat, target) {
            path.insert(0, ipat.idx);
            return Some(path);
        }
    }
    None
}

/// Analyze a match made of these arms (pattern, has_guard) and return the (arm index, subpattern
/// path) of subpatterns found redundant resp. useless.
fn check(
    ty: Ty,
    arms: &[(DeconstructedPat<Cx>, bool)],
) -> (Vec<(usize, Vec<usize>)>, Vec<(usize, Vec<usize>)>) {
    let arms: Vec<MatchArm<'_, Cx>> = arms
        .iter()
        .map(|(pat, has_guard)| MatchArm { pat, has_guard: *has_guard, arm_data: () })
        .collect();
    let report =
        compute_match_usefulness(&arms, ty, PlaceValidity::ValidOnly, usize::MAX, false).unwrap();
    let mut redundant = Vec::new();
    let mut useless = Vec::new();
    for (arm_id, (arm, usefulness)) in report.arm_usefulness.iter().enumerate() {
        if let Usefulness::Useful(redundant_subpats) = usefulness {
            for (pat, _) in redundant_subpats {
                redundant.push((arm_id, find_path(arm.pat, pat).unwrap()));
            }
        }
    }
    for (arm_id, pat, _) in &report.useless_subpatterns {
        let arm_pat = report.arm_usefulness[*arm_id].0.pat;
        useless.push((*arm_id, find_path(arm_pat, pat).unwrap()));
    }
    (redundant, useless)
}

#[track_caller]
fn assert_lints(
    ty: Ty,
    arms: &[(DeconstructedPat<Cx>, bool)],
    expected_redundant: &[(usize, &[usize])],
    expected_useless: &[(usize, &[usize])],
) {
    let (redundant, useless) = check(ty, arms);
    let redundant: Vec<_> = redundant.iter().map(|(i, path)| (*i, path.as_slice())).collect();
    let useless: Vec<_> = useless.iter().map(|(i, path)| (*i, path.as_slice())).collect();
    assert_eq!(redundant.as_slice(), expected_redundant, "redundant subpatterns mismatch");
    assert_eq!(useless.as_slice(), expected_useless, "useless subpatterns mismatch");
}

#[test]
fn test_useless_or_alternative() {
    let ty = Ty::U8;

    // `0 | _`: the `0` is useless (issue #160772).
    let pat = or_pat(ty, pats!(ty; 0, _));
    assert_lints(ty, &[(pat, false)], &[], &[(0, &[0])]);

    // `_ | 0`: the `0` is redundant (existing behavior), not additionally useless.
    let pat = or_pat(ty, pats!(ty; _, 0));
    assert_lints(ty, &[(pat, false)], &[(0, &[1])], &[]);

    // `0 | 0..`: the `0` is useless even though the other side isn't a wildcard.
    let pat = or_pat(ty, pats!(ty; 0, 0..));
    assert_lints(ty, &[(pat, false)], &[], &[(0, &[0])]);

    // `0..=1 | 1..=2 | 2..=3`: the other two collectively cover the middle alternative.
    let pat = or_pat(ty, pats!(ty; 0..=1, 1..=2, 2..=3));
    assert_lints(ty, &[(pat, false)], &[], &[(0, &[1])]);

    // `0..=1 | 1..=2`: both alternatives have values of their own.
    let pat = or_pat(ty, pats!(ty; 0..=1, 1..=2));
    assert_lints(ty, &[(pat, false)], &[], &[]);

    // `0..=1 | 0..=1`: the second is unreachable (existing behavior), the first useless.
    let pat = or_pat(ty, pats!(ty; 0..=1, 0..=1));
    assert_lints(ty, &[(pat, false)], &[(0, &[1])], &[(0, &[0])]);
}

#[test]
fn test_useless_with_earlier_arms() {
    let ty = Ty::U8;

    // `1` is redundant (covered on its left); `0..=1` is useless (earlier arm + sibling).
    let arm1 = pat!(ty; 0);
    let arm2 = or_pat(ty, pats!(ty; 0..=1, 1));
    assert_lints(ty, &[(arm1, false), (arm2, false)], &[(1, &[1])], &[(1, &[0])]);

    // `0 => ..` then `0 | 1 => ..`: `0` is redundant, and must not also be reported useless.
    let arm1 = pat!(ty; 0);
    let arm2 = or_pat(ty, pats!(ty; 0, 1));
    assert_lints(ty, &[(arm1, false), (arm2, false)], &[(1, &[0])], &[]);
}

#[test]
fn test_useless_with_guards() {
    let ty = Ty::U8;

    // `0` is useless but not redundant: the guard keeps it reachable.
    let arm1 = or_pat(ty, pats!(ty; 0, _));
    let arm2 = pat!(ty; _);
    assert_lints(ty, &[(arm1, true), (arm2, false)], &[], &[(0, &[0])]);

    // `0` in the second arm can match (when the guard fails) yet removing it changes nothing.
    let arm1 = pat!(ty; 0);
    let arm2 = or_pat(ty, pats!(ty; 0, _));
    assert_lints(ty, &[(arm1, true), (arm2, false)], &[], &[(1, &[0])]);
}

#[test]
fn test_useless_nested() {
    let ty = Ty::U8;
    let tuple_ty = Ty::Tuple(&[Ty::U8, Ty::Bool]);

    // `(0 | _, true)`: the `0` is useless.
    let inner_or = or_pat(ty, pats!(ty; 0, _));
    let true_pat = pat!(Ty::Bool; true);
    let arm1 = DeconstructedPat::new(
        Constructor::Struct,
        vec![inner_or.at_index(0), true_pat.at_index(1)],
        2,
        tuple_ty,
        (),
    );
    let arm2 = pat!(tuple_ty; _);
    assert_lints(tuple_ty, &[(arm1, false), (arm2, false)], &[], &[(0, &[0, 0])]);

    // `(0..=1 | 1..=2, 0 | 1)`: every alternative has a value only it brings to the arm.
    let or_a = or_pat(ty, pats!(ty; 0..=1, 1..=2));
    let or_b = or_pat(ty, pats!(ty; 0, 1));
    let arm1 = DeconstructedPat::new(
        Constructor::Struct,
        vec![or_a.at_index(0), or_b.at_index(1)],
        2,
        tuple_ty,
        (),
    );
    let arm2 = pat!(tuple_ty; _);
    assert_lints(tuple_ty, &[(arm1, false), (arm2, false)], &[], &[]);
}
