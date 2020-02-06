#![deny(clippy::panicking_unwrap, clippy::unnecessary_unwrap)]
#![allow(clippy::if_same_then_else)]

macro_rules! m {
    ($a:expr) => {
        if $a.is_some() {
            $a.unwrap(); // unnecessary
        }
    };
}

fn main() {
    let x = Some(());
    if x.is_some() {
        x.unwrap(); // unnecessary
    } else {
        x.unwrap(); // will panic
    }
    if x.is_none() {
        x.unwrap(); // will panic
    } else {
        x.unwrap(); // unnecessary
    }
    m!(x);
    let mut x: Result<(), ()> = Ok(());
    if x.is_ok() {
        x.unwrap(); // unnecessary
        x.unwrap_err(); // will panic
    } else {
        x.unwrap(); // will panic
        x.unwrap_err(); // unnecessary
    }
    if x.is_err() {
        x.unwrap(); // will panic
        x.unwrap_err(); // unnecessary
    } else {
        x.unwrap(); // unnecessary
        x.unwrap_err(); // will panic
    }
    if x.is_ok() {
        x = Err(());
        x.unwrap(); // not unnecessary because of mutation of x
                    // it will always panic but the lint is not smart enough to see this (it only
                    // checks if conditions).
    } else {
        x = Ok(());
        x.unwrap_err(); // not unnecessary because of mutation of x
                        // it will always panic but the lint is not smart enough to see this (it
                        // only checks if conditions).
    }

    assert!(x.is_ok(), "{:?}", x.unwrap_err()); // ok, it's a common test pattern
}
