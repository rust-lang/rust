#![warn(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::unnecessary_literal_unwrap)]
#![feature(never_type)]

use std::convert::Infallible;

trait OptionExt {
    type Item;

    fn unwrap_err(self) -> Self::Item;

    fn expect_err(self, msg: &str) -> Self::Item;
}

impl<T> OptionExt for Option<T> {
    type Item = T;
    fn unwrap_err(self) -> T {
        panic!();
    }

    fn expect_err(self, msg: &str) -> T {
        panic!();
    }
}

fn main() {
    Some(3).unwrap();
    //~^ ERROR: used `unwrap()` on an `Option` value
    Some(3).expect("Hello world!");
    //~^ ERROR: used `expect()` on an `Option` value

    // Don't trigger on unwrap_err on an option
    Some(3).unwrap_err();
    Some(3).expect_err("Hellow none!");

    // Issue #11245: The `Err` variant can never be constructed so do not lint this.
    let x: Result<(), !> = Ok(());
    x.unwrap();
    x.expect("is `!` (never)");
    let x: Result<(), Infallible> = Ok(());
    x.unwrap();
    x.expect("is never-like (0 variants)");

    let a: Result<i32, i32> = Ok(3);
    a.unwrap();
    //~^ ERROR: used `unwrap()` on a `Result` value
    a.expect("Hello world!");
    //~^ ERROR: used `expect()` on a `Result` value
    a.unwrap_err();
    //~^ ERROR: used `unwrap_err()` on a `Result` value
    a.expect_err("Hello error!");
    //~^ ERROR: used `expect_err()` on a `Result` value
}
