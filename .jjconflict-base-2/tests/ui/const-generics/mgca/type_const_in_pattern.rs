//@ check-pass
#![feature(gca_min_const_items)]
#![expect(incomplete_features)]
#![allow(irrefutable_let_patterns)]

use std::gca;

const CONST: usize = gca!(1_usize);

struct Inherent;

impl Inherent {
    const BAR: usize = gca!(1_usize);
}

trait Trait {
    #[rustc_always_gca]
    const BAZ: usize;
}

struct Assoc;

impl Trait for Assoc {
    const BAZ: usize = gca!(1_usize);
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
