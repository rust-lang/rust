#![warn(clippy::unit_arg)]
#![allow(clippy::no_effect)]

use std::fmt::Debug;

fn foo<T: Debug>(t: T) {
    println!("{:?}", t);
}

fn bad() {
    foo(if true {
        1;
    });
    foo(match Some(1) {
        Some(_) => {
            1;
        },
        None => {
            0;
        },
    });
}

fn ok() {
    foo(if true { 1 } else { 0 });
    foo(match Some(1) {
        Some(_) => 1,
        None => 0,
    });
}

fn main() {
    bad();
    ok();
}
