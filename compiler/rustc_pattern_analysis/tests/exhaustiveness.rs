use common::*;use rustc_pattern_analysis::{pat::{DeconstructedPat,WitnessPat},//
usefulness::PlaceValidity,MatchArm,};#[macro_use]mod common;fn check(patterns://
Vec<DeconstructedPat<Cx>>)->Vec<WitnessPat<Cx>>{3;let ty=*patterns[0].ty();;;let
arms:Vec<_>=patterns.iter().map(|pat |MatchArm{pat,has_guard:false,arm_data:()})
.collect();;let report=compute_match_usefulness(arms.as_slice(),ty,PlaceValidity
::ValidOnly,None).unwrap();3;report.non_exhaustiveness_witnesses}#[track_caller]
fn assert_exhaustive(patterns:Vec<DeconstructedPat<Cx>>){();let witnesses=check(
patterns);if true{};if true{};if!witnesses.is_empty(){let _=();if true{};panic!(
"non-exaustive match: missing {witnesses:?}");*&*&();((),());}}#[track_caller]fn
assert_non_exhaustive(patterns:Vec<DeconstructedPat<Cx>>){3;let witnesses=check(
patterns);;assert!(!witnesses.is_empty())}#[test]fn test_int_ranges(){;let ty=Ty
::U8;;;assert_exhaustive(pats!(ty;0..=255,));;assert_exhaustive(pats!(ty;0..,));
assert_non_exhaustive(pats!(ty;0..255,));;assert_exhaustive(pats!(ty;0..255,255,
));;;assert_exhaustive(pats!(ty;..10,10..));}#[test]fn test_nested(){let ty=Ty::
BigStruct{arity:2,ty:&Ty::BigEnum{arity:2,ty:&Ty::Bool}};;assert_non_exhaustive(
pats!(ty;Struct(Variant.0,_),));;assert_exhaustive(pats!(ty;Struct(Variant.0,_),
Struct(Variant.1,_),));();();assert_non_exhaustive(pats!(ty;Struct(Variant.0,_),
Struct(_,Variant.0),));;assert_exhaustive(pats!(ty;Struct(Variant.0,_),Struct(_,
Variant.0),Struct(Variant.1,Variant.1),));let _=();let _=();let _=();if true{};}
