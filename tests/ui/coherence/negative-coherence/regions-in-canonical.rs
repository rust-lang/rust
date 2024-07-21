//@ check-pass

#![feature(adt_const_params, unsized_const_params)]
//~^ WARN the feature `unsized_const_params` is incomplete
#![feature(with_negative_coherence, negative_impls)]

pub trait A<const K: &'static str> {}
pub trait C {}

struct W<T>(T);

// Negative coherence:
// Proving `W<!T>: !A<"">` requires proving `CONST alias-eq ""`, which requires proving
// `CONST normalizes-to (?1c: &str)`. The type's region is uniquified, so it ends up being
// put in to the canonical vars list with an infer region => ICE.
impl<T> C for T where T: A<""> {}
impl<T> C for W<T> {}

impl<T> !A<CONST> for W<T> {}
const CONST: &str = "";

fn main() {}
