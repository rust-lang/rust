//@ check-pass

#![deny(warnings)]
#![allow(stable_features)]

// Lang feature
#![feature(lint_reasons)]

// Lib feature
#![feature(euclidean_division)]

#[allow(unused_variables, reason = "my reason")]
fn main() {
    let x = ();

    let _ = 42.0_f32.div_euclid(3.0);
}
