//@compile-flags: -Znext-solver=globally
#![feature(gca_const_items, gca_min_const_items)]

struct S<const N: usize>;
fn foo<const N: usize>(_: S<{ const { const { N } } }>) {}
//~^ ERROR: generic parameters in const blocks are not allowed; use a named `const` item instead


fn main() {}
