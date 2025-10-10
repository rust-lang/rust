//@aux-build:proc_macros.rs
//@require-annotations-for-level: WARN
#![allow(clippy::no_effect, deprecated, unused)]
#![allow(clippy::legacy_numeric_constants)] // For imports.

#[macro_use]
extern crate proc_macros;

pub mod a {
    pub use std::u128;
}

macro_rules! b {
    () => {
        mod b {
            #[warn(clippy::legacy_numeric_constants)]
            fn b() {
                let x = std::u64::MAX;
                //~^ ERROR: usage of a legacy numeric constant
                //~| HELP: use the associated constant instead
            }
        }
    };
}

use std::u8::MIN;
use std::u32::MAX;
use std::{f64, u32};

#[warn(clippy::legacy_numeric_constants)]
fn main() {
    std::f32::EPSILON;
    //~^ ERROR: usage of a legacy numeric constant
    //~| HELP: use the associated constant instead
    std::u8::MIN;
    //~^ ERROR: usage of a legacy numeric constant
    //~| HELP: use the associated constant instead
    std::usize::MIN;
    //~^ ERROR: usage of a legacy numeric constant
    //~| HELP: use the associated constant instead
    std::u32::MAX;
    //~^ ERROR: usage of a legacy numeric constant
    //~| HELP: use the associated constant instead
    core::u32::MAX;
    //~^ ERROR: usage of a legacy numeric constant
    //~| HELP: use the associated constant instead
    MAX;
    //~^ ERROR: usage of a legacy numeric constant
    //~| HELP: use the associated constant instead
    i32::max_value();
    //~^ ERROR: usage of a legacy numeric method
    //~| HELP: use the associated constant instead
    u8::max_value();
    //~^ ERROR: usage of a legacy numeric method
    //~| HELP: use the associated constant instead
    u8::min_value();
    //~^ ERROR: usage of a legacy numeric method
    //~| HELP: use the associated constant instead
    ::std::u8::MIN;
    //~^ ERROR: usage of a legacy numeric constant
    //~| HELP: use the associated constant instead
    ::std::primitive::u8::min_value();
    //~^ ERROR: usage of a legacy numeric method
    //~| HELP: use the associated constant instead
    std::primitive::i32::max_value();
    //~^ ERROR: usage of a legacy numeric method
    //~| HELP: use the associated constant instead
    self::a::u128::MAX;
    //~^ ERROR: usage of a legacy numeric constant
    //~| HELP: use the associated constant instead
    u32::MAX;
    u128::MAX;
    f32::EPSILON;
    ::std::primitive::u8::MIN;
    std::f32::consts::E;
    f64::consts::E;
    u8::MIN;
    std::f32::consts::E;
    f64::consts::E;
    b!();

    <std::primitive::i32>::max_value();
    //~^ ERROR: usage of a legacy numeric method
    //~| HELP: use the associated constant instead
    [(0, "", std::i128::MAX)];
    //~^ ERROR: usage of a legacy numeric constant
    //~| HELP: use the associated constant instead
    (i32::max_value());
    //~^ ERROR: usage of a legacy numeric method
    //~| HELP: use the associated constant instead
    assert_eq!(0, -(i32::max_value()));
    //~^ ERROR: usage of a legacy numeric method
    //~| HELP: use the associated constant instead
    (std::i128::MAX);
    //~^ ERROR: usage of a legacy numeric constant
    //~| HELP: use the associated constant instead
    (<u32>::max_value());
    //~^ ERROR: usage of a legacy numeric method
    //~| HELP: use the associated constant instead
    ((i32::max_value)());
    //~^ ERROR: usage of a legacy numeric method
    //~| HELP: use the associated constant instead
    type Ω = i32;
    Ω::max_value();
    //~^ ERROR: usage of a legacy numeric method
    //~| HELP: use the associated constant instead
}

#[warn(clippy::legacy_numeric_constants)]
fn ext() {
    external! {
        ::std::primitive::u8::MIN;
        ::std::u8::MIN;
        ::std::primitive::u8::min_value();
        use std::u64;
        use std::u8::MIN;
    }
}

#[allow(clippy::legacy_numeric_constants)]
fn allow() {
    ::std::primitive::u8::MIN;
    ::std::u8::MIN;
    ::std::primitive::u8::min_value();
    use std::u8::MIN;
    use std::u64;
}

#[warn(clippy::legacy_numeric_constants)]
#[clippy::msrv = "1.42.0"]
fn msrv_too_low() {
    std::u32::MAX;
}

#[warn(clippy::legacy_numeric_constants)]
#[clippy::msrv = "1.43.0"]
fn msrv_juust_right() {
    std::u32::MAX;
    //~^ ERROR: usage of a legacy numeric constant
}
