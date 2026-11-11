//@ aux-build:trivial-cast-ice.rs
//@ check-pass

// Demonstrates the ICE in #102561

#![deny(trivial_casts)]

extern crate trivial_cast_ice;

fn main() {
    trivial_cast_ice::foo!();
}
