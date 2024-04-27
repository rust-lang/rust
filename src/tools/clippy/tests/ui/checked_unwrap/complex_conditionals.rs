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
        //~^ ERROR: called `unwrap` on `x` after checking its variant with `is_ok`
        // will panic
        x.unwrap_err();
        //~^ ERROR: this call to `unwrap_err()` will always panic
        // will panic
        y.unwrap();
        //~^ ERROR: this call to `unwrap()` will always panic
        // unnecessary
        y.unwrap_err();
        //~^ ERROR: called `unwrap_err` on `y` after checking its variant with `is_err`
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
        //~^ ERROR: this call to `unwrap()` will always panic
        // unnecessary
        x.unwrap_err();
        //~^ ERROR: called `unwrap_err` on `x` after checking its variant with `is_ok`
        // will panic
        y.unwrap();
        //~^ ERROR: this call to `unwrap()` will always panic
        // unnecessary
        y.unwrap_err();
        //~^ ERROR: called `unwrap_err` on `y` after checking its variant with `is_ok`
    }
    let z: Result<(), ()> = Ok(());
    if x.is_ok() && !(y.is_ok() || z.is_err()) {
        // unnecessary
        x.unwrap();
        //~^ ERROR: called `unwrap` on `x` after checking its variant with `is_ok`
        // will panic
        x.unwrap_err();
        //~^ ERROR: this call to `unwrap_err()` will always panic
        // will panic
        y.unwrap();
        //~^ ERROR: this call to `unwrap()` will always panic
        // unnecessary
        y.unwrap_err();
        //~^ ERROR: called `unwrap_err` on `y` after checking its variant with `is_ok`
        // unnecessary
        z.unwrap();
        //~^ ERROR: called `unwrap` on `z` after checking its variant with `is_err`
        // will panic
        z.unwrap_err();
        //~^ ERROR: this call to `unwrap_err()` will always panic
    }
    if x.is_ok() || !(y.is_ok() && z.is_err()) {
        // not statically determinable whether any of the following will always succeed or always fail:
        x.unwrap();
        y.unwrap();
        z.unwrap();
    } else {
        // will panic
        x.unwrap();
        //~^ ERROR: this call to `unwrap()` will always panic
        // unnecessary
        x.unwrap_err();
        //~^ ERROR: called `unwrap_err` on `x` after checking its variant with `is_ok`
        // unnecessary
        y.unwrap();
        //~^ ERROR: called `unwrap` on `y` after checking its variant with `is_ok`
        // will panic
        y.unwrap_err();
        //~^ ERROR: this call to `unwrap_err()` will always panic
        // will panic
        z.unwrap();
        //~^ ERROR: this call to `unwrap()` will always panic
        // unnecessary
        z.unwrap_err();
        //~^ ERROR: called `unwrap_err` on `z` after checking its variant with `is_err`
    }
}

fn main() {}
