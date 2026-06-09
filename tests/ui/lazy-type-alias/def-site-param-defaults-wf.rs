//! Ensure that we check generic parameter defaults for well-formedness at the definition site.
#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type Alias<T = Vec<str>, const N: usize = { 0 - 1 }> = T;
//~^ ERROR attempt to compute `0_usize - 1_usize`, which would overflow
//~| ERROR the size for values of type `str` cannot be known at compilation time

fn main() {}
