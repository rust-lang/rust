// compile-pass

#![feature(rustc_attrs)]

#[rustc_dummy(a b c d)]
#[rustc_dummy[a b c d]]
#[rustc_dummy{a b c d}]
fn main() {}
