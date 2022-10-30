// astconv uses `FreshTy(0)` as a dummy `Self` type when instanciating trait objects.
// This `FreshTy(0)` can leak into substs, causing ICEs in several places.
// Using `save-analysis` triggers type-checking `f` that would be normally skipped
// as `type_of` emitted an error.
//
// compile-flags: -Zsave-analysis

#![feature(trait_alias)]

pub trait SelfInput = Fn(&mut Self);

pub fn f(_f: &dyn SelfInput) {}
//~^ ERROR the trait alias `SelfInput` cannot be made into an object [E0038]

fn main() {}
