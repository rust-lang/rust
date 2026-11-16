//@ compile-flags: -Znext-solver
#![feature(gca_min_const_items, gca_const_items)]
#![expect(incomplete_features)]

fn foo<const N: usize>() {}
fn bar<const N: usize>() {
    foo::<const { N + 1 }>();
               //~^ ERROR: generic parameters in const blocks are not allowed; use a named `const` item instead
}
fn main(){}
