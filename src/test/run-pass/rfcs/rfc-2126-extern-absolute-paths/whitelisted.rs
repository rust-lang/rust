// run-pass
// edition:2018

// Tests that `core` and `std` are always available.
use core::iter;
use std::io;
// FIXME(eddyb) Add a `meta` crate to the distribution.
// use meta;

fn main() {
    for _ in iter::once(()) {
        io::stdout();
    }
}
