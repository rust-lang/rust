//@no-rustfix
//@require-annotations-for-level: WARN
//@aux-build:proc_macros.rs
#![allow(clippy::no_effect, deprecated, unused)]
#![warn(clippy::legacy_numeric_constants)]

#[macro_use]
extern crate proc_macros;

use std::u128 as _;
//~^ ERROR: importing legacy numeric constants
//~| HELP: remove this import
pub mod a {
    pub use std::{mem, u128};
    //~^ ERROR: importing legacy numeric constants
    //~| HELP: remove this import
}

macro_rules! b {
    () => {
        mod b {
            use std::u32;
            //~^ ERROR: importing legacy numeric constants
            //~| HELP: remove this import
        }
    };
}

fn main() {
    use std::u32::MAX;
    //~^ ERROR: importing a legacy numeric constant
    //~| HELP: remove this import and use the associated constant `u32::MAX`
    use std::u8::MIN;
    //~^ ERROR: importing a legacy numeric constant
    //~| HELP: remove this import and use the associated constant `u8::MIN`
    f64::MAX;
    use std::u32;
    //~^ ERROR: importing legacy numeric constants
    //~| HELP: remove this import
    u32::MAX;
    use std::f32::MIN_POSITIVE;
    //~^ ERROR: importing a legacy numeric constant
    //~| HELP: remove this import and use the associated constant `f32::MIN_POSITIVE`
    use std::f64;
    use std::i16::*;
    //~^ ERROR: importing legacy numeric constants
    //~| HELP: remove this import and use associated constants `i16::<CONST>`
    u128::MAX;
    f32::EPSILON;
    f64::EPSILON;
    ::std::primitive::u8::MIN;
    std::f32::consts::E;
    f64::consts::E;
    u8::MIN;
    std::f32::consts::E;
    f64::consts::E;
    b!();
}

fn ext() {
    external! {
        ::std::primitive::u8::MIN;
        ::std::u8::MIN;
        ::std::primitive::u8::min_value();
        use std::u64;
        use std::u8::MIN;
    }
}

#[clippy::msrv = "1.42.0"]
fn msrv_too_low() {
    use std::u32::MAX;
}

#[clippy::msrv = "1.43.0"]
fn msrv_juust_right() {
    use std::u32::MAX;
    //~^ ERROR: importing a legacy numeric constant
}
