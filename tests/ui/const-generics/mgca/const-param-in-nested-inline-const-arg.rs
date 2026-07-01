//@compile-flags: -Znext-solver=globally
#![feature(generic_const_args, min_generic_const_args)]

struct S<const N: usize>;
fn foo<const N: usize>(_: S<{ const { const { N } } }>) {}
//~^ ERROR: generic parameters in const blocks are not allowed; use a named `const` item instead


fn main() {}
