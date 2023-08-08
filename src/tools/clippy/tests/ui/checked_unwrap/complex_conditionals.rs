#![deny(clippy::panicking_unwrap, clippy::unnecessary_unwrap)]
#![allow(
    clippy::if_same_then_else,
    clippy::branches_sharing_code,
    clippy::unnecessary_literal_unwrap
)]

fn test_complex_conditions() {
    let x: Result<(), ()> = Ok(());
    let y: Result<(), ()> = Ok(());
    if x.is_ok() && y.is_err() {
        x.unwrap(); // unnecessary
        x.unwrap_err(); // will panic
        y.unwrap(); // will panic
        y.unwrap_err(); // unnecessary
    } else {
        // not statically determinable whether any of the following will always succeed or always fail:
        x.unwrap();
        x.unwrap_err();
        y.unwrap();
        y.unwrap_err();
    }

    if x.is_ok() || y.is_ok() {
        // not statically determinable whether any of the following will always succeed or always fail:
        x.unwrap();
        y.unwrap();
    } else {
        x.unwrap(); // will panic
        x.unwrap_err(); // unnecessary
        y.unwrap(); // will panic
        y.unwrap_err(); // unnecessary
    }
    let z: Result<(), ()> = Ok(());
    if x.is_ok() && !(y.is_ok() || z.is_err()) {
        x.unwrap(); // unnecessary
        x.unwrap_err(); // will panic
        y.unwrap(); // will panic
        y.unwrap_err(); // unnecessary
        z.unwrap(); // unnecessary
        z.unwrap_err(); // will panic
    }
    if x.is_ok() || !(y.is_ok() && z.is_err()) {
        // not statically determinable whether any of the following will always succeed or always fail:
        x.unwrap();
        y.unwrap();
        z.unwrap();
    } else {
        x.unwrap(); // will panic
        x.unwrap_err(); // unnecessary
        y.unwrap(); // unnecessary
        y.unwrap_err(); // will panic
        z.unwrap(); // will panic
        z.unwrap_err(); // unnecessary
    }
}

fn main() {}
