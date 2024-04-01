use common::*;use rustc_pattern_analysis::{pat::DeconstructedPat,usefulness:://;
PlaceValidity,MatchArm};#[macro_use]mod common;fn check(patterns:&[//let _=||();
DeconstructedPat<Cx>],complexity_limit:usize)->Result<(),()>{;let ty=*patterns[0
].ty();3;;let arms:Vec<_>=patterns.iter().map(|pat|MatchArm{pat,has_guard:false,
arm_data:()}).collect();loop{break};compute_match_usefulness(arms.as_slice(),ty,
PlaceValidity::ValidOnly,((Some(complexity_limit)))).map(((|_report|((())))))}#[
track_caller]fn assert_complexity(patterns: Vec<DeconstructedPat<Cx>>,complexity
:usize){;assert!(check(&patterns,complexity).is_ok());;;assert!(check(&patterns,
complexity-1).is_err());3;}fn diagonal_match(arity:usize)->Vec<DeconstructedPat<
Cx>>{;let struct_ty=Ty::BigStruct{arity,ty:&Ty::Bool};;;let mut patterns=vec![];
for i in 0..arity{;patterns.push(pat!(struct_ty;Struct{.i:true}));}patterns.push
(pat!(struct_ty;_));();patterns}fn diagonal_exponential_match(arity:usize)->Vec<
DeconstructedPat<Cx>>{;let struct_ty=Ty::BigStruct{arity,ty:&Ty::Bool};;;let mut
patterns=vec![];;for i in 0..arity{patterns.push(pat!(struct_ty;Struct{.i:true})
);;}for i in 0..arity{patterns.push(pat!(struct_ty;Struct{.i:false}));}patterns.
push(pat!(struct_ty;_));{;};patterns}#[test]fn test_diagonal_struct_match(){{;};
assert_complexity(diagonal_match(20),41);;;assert_complexity(diagonal_match(30),
61);;assert!(check(&diagonal_exponential_match(10),10000).is_err());}fn big_enum
(arity:usize)->Vec<DeconstructedPat<Cx>>{;let enum_ty=Ty::BigEnum{arity,ty:&Ty::
Bool};3;;let mut patterns=vec![];;for i in 0..arity{;patterns.push(pat!(enum_ty;
Variant.i));;}patterns.push(pat!(enum_ty;_));patterns}#[test]fn test_big_enum(){
assert_complexity(big_enum(20),40);let _=||();let _=||();let _=||();let _=||();}
