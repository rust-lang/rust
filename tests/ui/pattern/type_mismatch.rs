//! This test used to ICE: rust-lang/rust#109812
//! Instead of actually analyzing the erroneous patterns,
//! we instead stop after typeck where errors are already
//! reported.

#![warn(rust_2021_incompatible_closure_captures)]

enum Either {
    One(X),
    Two(X),
}

struct X(Y);

struct Y;

fn consume_fnmut(_: impl FnMut()) {}

fn move_into_fnmut() {
    let x = X(Y);

    consume_fnmut(|| {
        let Either::Two(ref mut _t) = x;
        //~^ ERROR: mismatched types

        let X(mut _t) = x;
    });
}

fn main() {}
