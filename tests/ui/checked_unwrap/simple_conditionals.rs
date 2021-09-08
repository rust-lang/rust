#![deny(clippy::panicking_unwrap, clippy::unnecessary_unwrap)]
#![allow(clippy::if_same_then_else, clippy::branches_sharing_code)]

macro_rules! m {
    ($a:expr) => {
        if $a.is_some() {
            $a.unwrap(); // unnecessary
        }
    };
}

macro_rules! checks_in_param {
    ($a:expr, $b:expr) => {
        if $a {
            $b;
        }
    };
}

macro_rules! checks_unwrap {
    ($a:expr, $b:expr) => {
        if $a.is_some() {
            $b;
        }
    };
}

macro_rules! checks_some {
    ($a:expr, $b:expr) => {
        if $a {
            $b.unwrap();
        }
    };
}

fn main() {
    let x = Some(());
    if x.is_some() {
        x.unwrap(); // unnecessary
        x.expect("an error message"); // unnecessary
    } else {
        x.unwrap(); // will panic
        x.expect("an error message"); // will panic
    }
    if x.is_none() {
        x.unwrap(); // will panic
    } else {
        x.unwrap(); // unnecessary
    }
    m!(x);
    checks_in_param!(x.is_some(), x.unwrap()); // ok
    checks_unwrap!(x, x.unwrap()); // ok
    checks_some!(x.is_some(), x); // ok
    let mut x: Result<(), ()> = Ok(());
    if x.is_ok() {
        x.unwrap(); // unnecessary
        x.expect("an error message"); // unnecessary
        x.unwrap_err(); // will panic
    } else {
        x.unwrap(); // will panic
        x.expect("an error message"); // will panic
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
        // not unnecessary because of mutation of x
        // it will always panic but the lint is not smart enough to see this (it only
        // checks if conditions).
        x.unwrap();
    } else {
        x = Ok(());
        // not unnecessary because of mutation of x
        // it will always panic but the lint is not smart enough to see this (it
        // only checks if conditions).
        x.unwrap_err();
    }

    assert!(x.is_ok(), "{:?}", x.unwrap_err()); // ok, it's a common test pattern
}
