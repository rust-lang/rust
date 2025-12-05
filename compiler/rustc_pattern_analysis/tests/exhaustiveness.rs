//! Test exhaustiveness checking.

#![allow(unused_crate_dependencies)]

use common::*;
use rustc_pattern_analysis::MatchArm;
use rustc_pattern_analysis::pat::{DeconstructedPat, WitnessPat};
use rustc_pattern_analysis::usefulness::PlaceValidity;

#[macro_use]
mod common;

/// Analyze a match made of these patterns.
fn run(
    ty: Ty,
    patterns: Vec<DeconstructedPat<Cx>>,
    exhaustive_witnesses: bool,
) -> Vec<WitnessPat<Cx>> {
    let arms: Vec<_> =
        patterns.iter().map(|pat| MatchArm { pat, has_guard: false, arm_data: () }).collect();
    let report = compute_match_usefulness(
        arms.as_slice(),
        ty,
        PlaceValidity::ValidOnly,
        usize::MAX,
        exhaustive_witnesses,
    )
    .unwrap();
    report.non_exhaustiveness_witnesses
}

/// Analyze a match made of these patterns. Panics if there are no patterns
fn check(patterns: Vec<DeconstructedPat<Cx>>) -> Vec<WitnessPat<Cx>> {
    let ty = *patterns[0].ty();
    run(ty, patterns, true)
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

use WhichWitnesses::*;
enum WhichWitnesses {
    AllOfThem,
    OnlySome,
}

#[track_caller]
/// We take the type as input to support empty matches.
fn assert_witnesses(
    which: WhichWitnesses,
    ty: Ty,
    patterns: Vec<DeconstructedPat<Cx>>,
    expected: Vec<&str>,
) {
    let exhaustive_wit = matches!(which, AllOfThem);
    let witnesses = run(ty, patterns, exhaustive_wit);
    let witnesses: Vec<_> = witnesses.iter().map(|w| format!("{w:?}")).collect();
    assert_eq!(witnesses, expected)
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
    // enum E { A(bool), B(bool) }
    // ty = (E, E)
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
fn test_witnesses() {
    // TY = Option<bool>
    const TY: Ty = Ty::Enum(&[Ty::Bool, UNIT]);
    // ty = (Option<bool>, Option<bool>)
    let ty = Ty::Tuple(&[TY, TY]);
    assert_witnesses(AllOfThem, ty, vec![], vec!["(_, _)"]);
    assert_witnesses(
        OnlySome,
        ty,
        pats!(ty;
            (Variant.0(false), Variant.0(false)),
        ),
        vec!["(Enum::Variant1(_), _)"],
    );
    assert_witnesses(
        AllOfThem,
        ty,
        pats!(ty;
            (Variant.0(false), Variant.0(false)),
        ),
        vec![
            "(Enum::Variant0(false), Enum::Variant0(true))",
            "(Enum::Variant0(false), Enum::Variant1(_))",
            "(Enum::Variant0(true), _)",
            "(Enum::Variant1(_), _)",
        ],
    );
    assert_witnesses(
        OnlySome,
        ty,
        pats!(ty;
            (_, Variant.0(false)),
        ),
        vec!["(_, Enum::Variant1(_))"],
    );
    assert_witnesses(
        AllOfThem,
        ty,
        pats!(ty;
            (_, Variant.0(false)),
        ),
        vec!["(_, Enum::Variant0(true))", "(_, Enum::Variant1(_))"],
    );

    let ty = Ty::NonExhaustiveEnum(&[UNIT, UNIT, UNIT]);
    assert_witnesses(
        OnlySome,
        ty,
        pats!(ty;
            Variant.0,
        ),
        vec!["_"],
    );
    assert_witnesses(
        AllOfThem,
        ty,
        pats!(ty;
            Variant.0,
        ),
        vec!["Enum::Variant1(_)", "Enum::Variant2(_)", "_"],
    );

    // Assert we put `true` before `false`.
    assert_witnesses(AllOfThem, Ty::Bool, Vec::new(), vec!["true", "false"]);
}

#[test]
fn test_empty() {
    // `TY = Result<bool, !>`
    const TY: Ty = Ty::Enum(&[Ty::Bool, NEVER]);
    assert_exhaustive(pats!(TY;
        Variant.0,
    ));
    let ty = Ty::Tuple(&[Ty::Bool, TY]);
    assert_exhaustive(pats!(ty;
        (true, Variant.0),
        (false, Variant.0),
    ));
}
