//@ check-fail

#![feature(extern_item_impls)]

#[eii(eii1)]
//~^ ERROR `eii1` has more than one default implementation which is not supported
fn a() {}

#[eii(eii1)]
//~^ ERROR the name `eii1` is defined multiple times
//~| ERROR `#[eii1]` required, but not found
fn main() {}
//~^ ERROR the `main` function cannot be declared in an `extern` block
