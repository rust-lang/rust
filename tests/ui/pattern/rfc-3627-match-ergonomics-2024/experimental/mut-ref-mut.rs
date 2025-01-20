//@ edition: 2024
//@ revisions: classic structural
//! Test diagnostics for binding with `mut` when the default binding mode is by-ref.
#![allow(incomplete_features)]
#![cfg_attr(classic, feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(structural, feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    struct Foo(u8);

    let Foo(mut a) = &Foo(0);
    //~^ ERROR: binding cannot be both mutable and by-reference
    a = &42;

    let Foo(mut a) = &mut Foo(0);
    //~^ ERROR: binding cannot be both mutable and by-reference
    a = &mut 42;
}
