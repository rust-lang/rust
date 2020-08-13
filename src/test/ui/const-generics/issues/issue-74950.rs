// [full] build-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]


#[derive(PartialEq, Eq)]
struct Inner;

// Note: We emit the error 5 times if we don't deduplicate:
// - struct definition
// - impl PartialEq
// - impl Eq
// - impl StructuralPartialEq
// - impl StructuralEq
#[derive(PartialEq, Eq)]
struct Outer<const I: Inner>;
//[min]~^ using `Inner` as const generic parameters is forbidden
//[min]~| using `Inner` as const generic parameters is forbidden
//[min]~| using `Inner` as const generic parameters is forbidden
//[min]~| using `Inner` as const generic parameters is forbidden
//[min]~| using `Inner` as const generic parameters is forbidden

fn main() {}
