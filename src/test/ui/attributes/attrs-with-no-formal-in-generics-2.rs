// This test checks variations on `<#[attr] 'a, #[oops]>`, where
// `#[oops]` is left dangling (that is, it is unattached, with no
// formal binding following it).

#![feature(rustc_attrs)]

struct RefAny<'a, T>(&'a T);

impl<#[rustc_dummy] 'a, #[rustc_dummy] T, #[oops]> RefAny<'a, T> {}
//~^ ERROR trailing attribute after generic parameter

fn main() {}
