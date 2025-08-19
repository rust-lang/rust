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
    //~^ unwrap_used

    Some(3).expect("Hello world!");
    //~^ expect_used

    // Don't trigger on unwrap_err on an option
    Some(3).unwrap_err();
    Some(3).expect_err("Hello none!");

    // Issue #11245: The `Err` variant can never be constructed so do not lint this.
    let x: Result<(), !> = Ok(());
    x.unwrap();
    x.expect("is `!` (never)");
    let x: Result<(), Infallible> = Ok(());
    x.unwrap();
    x.expect("is never-like (0 variants)");

    let a: Result<i32, i32> = Ok(3);
    a.unwrap();
    //~^ unwrap_used

    a.expect("Hello world!");
    //~^ expect_used

    a.unwrap_err();
    //~^ unwrap_used

    a.expect_err("Hello error!");
    //~^ expect_used

    // Don't trigger in compile time contexts by default
    const SOME: Option<i32> = Some(3);
    const UNWRAPPED: i32 = SOME.unwrap();
    const EXPECTED: i32 = SOME.expect("Not three?");
    const {
        SOME.unwrap();
    }
    const {
        SOME.expect("Still not three?");
    }
}

mod with_expansion {
    macro_rules! open {
        ($file:expr) => {
            std::fs::File::open($file)
        };
    }

    fn test(file: &str) {
        use std::io::Read;
        let mut s = String::new();
        let _ = open!(file).unwrap(); //~ unwrap_used
        let _ = open!(file).expect("can open"); //~ expect_used
        let _ = open!(file).unwrap_err(); //~ unwrap_used
        let _ = open!(file).expect_err("can open"); //~ expect_used
    }
}
