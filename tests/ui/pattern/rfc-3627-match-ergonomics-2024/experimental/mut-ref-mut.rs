//@ edition: 2024
//@ revisions: classic2024 structural2024
//! Test diagnostics for binding with `mut` when the default binding mode is by-ref.
#![allow(incomplete_features)]
#![cfg_attr(classic2024, feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(structural2024, feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    struct Foo(u8);

    let Foo(mut a) = &Foo(0);
    //~^ ERROR: binding cannot be both mutable and by-reference
    a = &42;

    let Foo(mut a) = &mut Foo(0);
    //~^ ERROR: binding cannot be both mutable and by-reference
    a = &mut 42;
}
