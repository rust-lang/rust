//@ check-pass
// This test should compile without an ICE.
#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

#[type_const]
const CONST: usize = 1;

fn uses_const() {
    CONST;
}

fn main() {}
