// run-pass

// FIXME(centril): enforce the warnings and such somehow.

#![warn(warnings)]

#![allow(illegal_floating_point_literal_pattern)]

fn main() {
    if let 0.0 = 0.0 {}
}
