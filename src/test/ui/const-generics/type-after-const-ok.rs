// [full] run-pass
// revisions: full min
// Verifies that having generic parameters after constants is permitted
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

#[allow(dead_code)]
struct A<const N: usize, T>(T);
//[min]~^ ERROR type parameters must be declared prior to const parameters

fn main() {}
