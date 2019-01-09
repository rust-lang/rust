// Test that structural match is only permitted with a feature gate,
// and that if a feature gate is supplied, it permits the type to be
// used in a match.

// revisions: with_gate no_gate

// gate-test-structural_match

#![allow(unused)]
#![feature(rustc_attrs)]
#![cfg_attr(with_gate, feature(structural_match))]

#[structural_match] //[no_gate]~ ERROR semantics of constant patterns is not yet settled
struct Foo {
    x: u32
}

const FOO: Foo = Foo { x: 0 };

#[rustc_error]
fn main() { //[with_gate]~ ERROR compilation successful
    let y = Foo { x: 1 };
    match y {
        FOO => { }
        _ => { }
    }
}
