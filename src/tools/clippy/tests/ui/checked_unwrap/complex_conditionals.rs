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
        // unnecessary
        x.unwrap();
        //~^ unnecessary_unwrap

        // will panic
        x.unwrap_err();
        //~^ panicking_unwrap

        // will panic
        y.unwrap();
        //~^ panicking_unwrap

        // unnecessary
        y.unwrap_err();
        //~^ unnecessary_unwrap
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
        // will panic
        x.unwrap();
        //~^ panicking_unwrap

        // unnecessary
        x.unwrap_err();
        //~^ unnecessary_unwrap

        // will panic
        y.unwrap();
        //~^ panicking_unwrap

        // unnecessary
        y.unwrap_err();
        //~^ unnecessary_unwrap
    }
    let z: Result<(), ()> = Ok(());
    if x.is_ok() && !(y.is_ok() || z.is_err()) {
        // unnecessary
        x.unwrap();
        //~^ unnecessary_unwrap

        // will panic
        x.unwrap_err();
        //~^ panicking_unwrap

        // will panic
        y.unwrap();
        //~^ panicking_unwrap

        // unnecessary
        y.unwrap_err();
        //~^ unnecessary_unwrap

        // unnecessary
        z.unwrap();
        //~^ unnecessary_unwrap

        // will panic
        z.unwrap_err();
        //~^ panicking_unwrap
    }
    if x.is_ok() || !(y.is_ok() && z.is_err()) {
        // not statically determinable whether any of the following will always succeed or always fail:
        x.unwrap();
        y.unwrap();
        z.unwrap();
    } else {
        // will panic
        x.unwrap();
        //~^ panicking_unwrap

        // unnecessary
        x.unwrap_err();
        //~^ unnecessary_unwrap

        // unnecessary
        y.unwrap();
        //~^ unnecessary_unwrap

        // will panic
        y.unwrap_err();
        //~^ panicking_unwrap

        // will panic
        z.unwrap();
        //~^ panicking_unwrap

        // unnecessary
        z.unwrap_err();
        //~^ unnecessary_unwrap
    }
}

fn main() {}
