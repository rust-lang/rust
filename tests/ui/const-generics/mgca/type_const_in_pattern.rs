//@ check-pass
#![feature(min_generic_const_args)]
#![expect(incomplete_features)]
#![allow(irrefutable_let_patterns)]

type const CONST: usize = 1_usize;

struct Inherent;

impl Inherent {
    type const BAR: usize = 1_usize;
}

trait Trait {
    type const BAZ: usize;
}

struct Assoc;

impl Trait for Assoc {
    type const BAZ: usize = 1_usize;
}

fn main() {
    if let CONST = 1 {}
    if let Inherent::BAR = 1 {}
    if let <Assoc as Trait>::BAZ = 1 {}

    match CONST {
        CONST => 0,
        _ => 1,
    };
}
