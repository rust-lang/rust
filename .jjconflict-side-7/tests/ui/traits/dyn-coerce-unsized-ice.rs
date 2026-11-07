// Regression test for:
// https://github.com/rust-lang/rust/issues/149094#issuecomment-4191071539
#![feature(coerce_unsized, unsized_fn_params)]
#![expect(internal_features)]

use std::ops::CoerceUnsized;

pub trait Trait {}

pub fn foo(x: dyn CoerceUnsized<*const dyn Trait>) -> *const dyn Trait {
//~^ ERROR: the trait `CoerceUnsized` is not dyn compatible [E0038]
    x
//~^ ERROR: the size for values of type `(dyn CoerceUnsized<*const (dyn Trait + 'static)> + 'static)` cannot be known at compilation time [E0277]
}

fn main() {}
