#![feature(rustc_attrs)]

#[rustc_legacy_const_generics(0)] //~ ERROR #[rustc_legacy_const_generics] must have one index for
fn foo1() {}

#[rustc_legacy_const_generics(1)] //~ ERROR index exceeds number of arguments
fn foo2<const X: usize>() {}

#[rustc_legacy_const_generics(2)] //~ ERROR index exceeds number of arguments
fn foo3<const X: usize>(_: u8) {}

#[rustc_legacy_const_generics(a)] //~ ERROR arguments should be non-negative integers
fn foo4<const X: usize>() {}

#[rustc_legacy_const_generics(1, a, 2, b)] //~ ERROR arguments should be non-negative integers
fn foo5<const X: usize, const Y: usize, const Z: usize, const W: usize>() {}

#[rustc_legacy_const_generics(0)] //~ ERROR attribute should be applied to a function
struct S;

#[rustc_legacy_const_generics(0usize)] //~ ERROR suffixed literals are not allowed in attributes
fn foo6<const X: usize>() {}

extern "C" {
    #[rustc_legacy_const_generics(1)] //~ ERROR attribute should be applied to a function
    fn foo7<const X: usize>(); //~ ERROR foreign items may not have const parameters
}

#[rustc_legacy_const_generics(0)] //~ ERROR #[rustc_legacy_const_generics] functions must only have
fn foo8<X>() {}

impl S {
    #[rustc_legacy_const_generics(0)] //~ ERROR attribute should be applied to a function
    fn foo9<const X: usize>() {}
}

#[rustc_legacy_const_generics] //~ ERROR malformed `rustc_legacy_const_generics` attribute
fn bar1() {}

#[rustc_legacy_const_generics = 1] //~ ERROR malformed `rustc_legacy_const_generics` attribute
fn bar2() {}

fn main() {}
