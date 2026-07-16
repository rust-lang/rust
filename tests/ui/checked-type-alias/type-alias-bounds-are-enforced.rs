// Check that we don't issue the lint `type_alias_bounds` for
// free type aliases since the bounds are indeed enforced.

//@ check-pass

#![feature(checked_type_aliases)]
#![allow(incomplete_features)]
#![deny(type_alias_bounds)]

use std::ops::Mul;

type Alias<T: Mul> = <T as Mul>::Output;

fn main() {}
