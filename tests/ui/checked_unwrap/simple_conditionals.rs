//@no-rustfix: overlapping suggestions
#![feature(lint_reasons)]
#![deny(clippy::panicking_unwrap, clippy::unnecessary_unwrap)]
#![allow(
    clippy::if_same_then_else,
    clippy::branches_sharing_code,
    clippy::unnecessary_literal_unwrap
)]

macro_rules! m {
    ($a:expr) => {
        if $a.is_some() {
            // unnecessary
            $a.unwrap();
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
        // unnecessary
        x.unwrap();
        // unnecessary
        x.expect("an error message");
    } else {
        // will panic
        x.unwrap();
        // will panic
        x.expect("an error message");
    }
    if x.is_none() {
        // will panic
        x.unwrap();
    } else {
        // unnecessary
        x.unwrap();
    }
    m!(x);
    // ok
    checks_in_param!(x.is_some(), x.unwrap());
    // ok
    checks_unwrap!(x, x.unwrap());
    // ok
    checks_some!(x.is_some(), x);
    let mut x: Result<(), ()> = Ok(());
    if x.is_ok() {
        // unnecessary
        x.unwrap();
        // unnecessary
        x.expect("an error message");
        // will panic
        x.unwrap_err();
    } else {
        // will panic
        x.unwrap();
        // will panic
        x.expect("an error message");
        // unnecessary
        x.unwrap_err();
    }
    if x.is_err() {
        // will panic
        x.unwrap();
        // unnecessary
        x.unwrap_err();
    } else {
        // unnecessary
        x.unwrap();
        // will panic
        x.unwrap_err();
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

    // ok, it's a common test pattern
    assert!(x.is_ok(), "{:?}", x.unwrap_err());
}

fn check_expect() {
    let x = Some(());
    if x.is_some() {
        #[expect(clippy::unnecessary_unwrap)]
        // unnecessary
        x.unwrap();
        #[expect(clippy::unnecessary_unwrap)]
        // unnecessary
        x.expect("an error message");
    } else {
        #[expect(clippy::panicking_unwrap)]
        // will panic
        x.unwrap();
        #[expect(clippy::panicking_unwrap)]
        // will panic
        x.expect("an error message");
    }
}
