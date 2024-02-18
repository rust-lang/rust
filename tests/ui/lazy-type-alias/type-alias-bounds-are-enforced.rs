// Check that we don't issue the lint `type_alias_bounds` for
// lazy type aliases since the bounds are indeed enforced.

//@ check-pass

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]
#![deny(type_alias_bounds)]

use std::ops::Mul;

type Alias<T: Mul> = <T as Mul>::Output;

fn main() {}
