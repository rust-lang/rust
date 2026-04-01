// HIR ty lowering uses `FreshTy(0)` as a dummy `Self` type when instanciating trait objects.
// This `FreshTy(0)` can leak into substs, causing ICEs in several places.

#![feature(trait_alias)]

pub trait SelfInput = Fn(&mut Self);

pub fn f(_f: &dyn SelfInput) {}
//~^ ERROR the trait alias `SelfInput` is not dyn compatible [E0038]

fn main() {}
