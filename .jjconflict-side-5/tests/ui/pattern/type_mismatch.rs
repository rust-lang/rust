//! These tests used to ICE: rust-lang/rust#109812, rust-lang/rust#150507
//! Instead of actually analyzing the erroneous patterns,
//! we instead stop after typeck where errors are already
//! reported.

#![warn(rust_2021_incompatible_closure_captures)]

enum Either {
    One(X),
    Two(X),
    Three { a: X },
}

struct X(Y);

struct Y;

struct Z(*const i32);
unsafe impl Send for Z {}

enum Meow {
    A { a: Z },
    B(Z),
}

fn consume_fnmut(_: impl FnMut()) {}

fn move_into_fnmut() {
    let x = X(Y);

    consume_fnmut(|| {
        let Either::Two(ref mut _t) = x;
        //~^ ERROR: mismatched types

        let X(mut _t) = x;
    });

    consume_fnmut(|| {
        let Either::Three { a: ref mut _t } = x;
        //~^ ERROR: mismatched types

        let X(mut _t) = x;
    });
}

fn tuple_against_array() {
    let variant: [();1] = [()];

    || match variant {
        (2,) => (),
        //~^ ERROR: mismatched types
        _ => {}
    };

    || {
        let ((2,) | _) = variant;
        //~^ ERROR: mismatched types
    };
}

// Reproducer that triggers the compatibility lint more reliably, instead of relying on the fact
// that at the time of writing, an unresolved integer type variable does not implement any
// auto-traits.
//
// The @_ makes this example also reproduce ICE #150507 before PR #138961
fn arcane() {
    let variant: [();1] = [()];

    || {
        match variant {
            (Z(y@_),) => {}
            //~^ ERROR: mismatched types
        }
    };

    || {
        match variant {
            Meow::A { a: Z(y@_) } => {}
            //~^ ERROR: mismatched types
        }
    };

    || {
        match variant {
            Meow::B(Z(y@_)) => {}
            //~^ ERROR: mismatched types
        }
    };
}

fn main() {}
