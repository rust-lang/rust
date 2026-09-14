//@ check-pass
#![feature(min_generic_const_args)]
#![expect(incomplete_features)]
#![allow(irrefutable_let_patterns)]

const CONST: usize = core::direct_const_arg!(1_usize);

struct Inherent;

impl Inherent {
    const BAR: usize = core::direct_const_arg!(1_usize);
}

trait Trait {
    #[rustc_always_gca]
    const BAZ: usize;
}

struct Assoc;

impl Trait for Assoc {
    const BAZ: usize = core::direct_const_arg!(1_usize);
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
