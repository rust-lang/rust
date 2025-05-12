//@no-rustfix: overlapping suggestions
//@aux-build:proc_macros.rs
#![allow(clippy::needless_if, unused)]
#![warn(clippy::manual_is_infinite, clippy::manual_is_finite)]

// FIXME(f16_f128): add tests for these types once constants are available

#[macro_use]
extern crate proc_macros;

const INFINITE: f32 = f32::INFINITY;
const NEG_INFINITE: f32 = f32::NEG_INFINITY;

fn fn_test() -> f64 {
    f64::NEG_INFINITY
}

fn fn_test_not_inf() -> f64 {
    112.0
}

fn main() {
    let x = 1.0f32;
    if x == f32::INFINITY || x == f32::NEG_INFINITY {}
    //~^ manual_is_infinite
    if x != f32::INFINITY && x != f32::NEG_INFINITY {}
    //~^ manual_is_finite
    if x == INFINITE || x == NEG_INFINITE {}
    //~^ manual_is_infinite
    if x != INFINITE && x != NEG_INFINITE {}
    //~^ manual_is_finite
    let x = 1.0f64;
    if x == f64::INFINITY || x == f64::NEG_INFINITY {}
    //~^ manual_is_infinite
    if x != f64::INFINITY && x != f64::NEG_INFINITY {}
    //~^ manual_is_finite
    // Don't lint
    if x.is_infinite() {}
    if x.is_finite() {}
    if x.abs() < f64::INFINITY {}
    if f64::INFINITY > x.abs() {}
    if f64::abs(x) < f64::INFINITY {}
    if f64::INFINITY > f64::abs(x) {}
    // Is not evaluated by `clippy_utils::constant`
    if x != f64::INFINITY && x != fn_test() {}
    // Not -inf
    if x != f64::INFINITY && x != fn_test_not_inf() {}
    const {
        let x = 1.0f64;
        if x == f64::INFINITY || x == f64::NEG_INFINITY {}
        //~^ manual_is_infinite
    }
    const X: f64 = 1.0f64;
    if const { X == f64::INFINITY || X == f64::NEG_INFINITY } {}
    if const { X != f64::INFINITY && X != f64::NEG_INFINITY } {}
    external! {
        let x = 1.0;
        if x == f32::INFINITY || x == f32::NEG_INFINITY {}
        if x != f32::INFINITY && x != f32::NEG_INFINITY {}
    }
    with_span! {
        span
        let x = 1.0;
        if x == f32::INFINITY || x == f32::NEG_INFINITY {}
        if x != f32::INFINITY && x != f32::NEG_INFINITY {}
    }
}
