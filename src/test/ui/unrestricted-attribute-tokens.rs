// build-pass (FIXME(62277): could be check-pass?)

#![feature(rustc_attrs)]

#[rustc_dummy(a b c d)]
#[rustc_dummy[a b c d]]
#[rustc_dummy{a b c d}]
fn main() {}
