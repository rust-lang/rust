// This used to ICE: #137188
// The missing parameter list on `N` was set to
// "infer from use site" in ast lowering, which
// caused later code to not emit a missing generic
// param error. The missing param was then attempted
// to be inferred, but inference of generic params
// is only possible within bodies. So a delayed
// bug was generated with no error ever reported.

#![feature(min_generic_const_args)]
#![allow(incomplete_features)]
trait Trait {}
impl Trait for [(); N] {}
//~^ ERROR: missing generics for function `N`
fn N<T>() {}
pub fn main() {}
