// [full] build-pass
// revisions: full min
#![cfg_attr(full, feature(adt_const_params))]
#![cfg_attr(full, allow(incomplete_features))]

#[cfg(full)]
use std::marker::ConstParamTy;

#[derive(PartialEq, Eq)]
#[cfg_attr(full, derive(ConstParamTy))]
struct Inner;

// Note: We emit the error 5 times if we don't deduplicate:
// - struct definition
// - impl PartialEq
// - impl Eq
// - impl StructuralPartialEq
// - impl StructuralEq
#[derive(PartialEq, Eq)]
struct Outer<const I: Inner>;
//[min]~^ `Inner` is forbidden
//[min]~| `Inner` is forbidden
//[min]~| `Inner` is forbidden
//[min]~| `Inner` is forbidden
//[min]~| `Inner` is forbidden

fn main() {}
