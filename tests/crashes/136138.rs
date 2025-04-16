//@ known-bug: #136138
#![feature(min_generic_const_args)]
struct U;
struct S<const N: U>()
where
    S<{ U }>:;
fn main() {}
