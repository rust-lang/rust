//! Regression test for https://github.com/rust-lang/rust/issues/129966
//!
//! Ensure we provide suggestion for wrongly ordered format parameters.

//@ run-rustfix
#![allow(dead_code)]

#[derive(Debug)]
struct Foo(u8, u8);

fn main() {
    let f = Foo(1, 2);

    println!("{f:?#}");
    //~^ ERROR invalid format string: expected `}`, found `#`
    //~| HELP did you mean `#?`?

    println!("{f:?x}");
    //~^ ERROR invalid format string: expected `}`, found `x`
    //~| HELP did you mean `x?`?

    println!("{f:?X}");
    //~^ ERROR invalid format string: expected `}`, found `X`
    //~| HELP did you mean `X?`?
}
