// This test is checking that you cannot override a `forbid` by adding in other
// attributes later in the same scope. (We already ensure that you cannot
// override it in nested scopes).

// If you turn off deduplicate diagnostics (which rustc turns on by default but
// compiletest turns off when it runs ui tests), then the errors are
// (unfortunately) repeated here because the checking is done as we read in the
// errors, and currently that happens two or three different times, depending on
// compiler flags.
//
// I decided avoiding the redundant output was not worth the time in engineering
// effort for bug like this, which 1. end users are unlikely to run into in the
// first place, and 2. they won't see the redundant output anyway.

// compile-flags: -Z deduplicate-diagnostics=yes

#![forbid(forbidden_lint_groups)]

fn forbid_first(num: i32) -> i32 {
    #![forbid(unused)]
    #![deny(unused)]
    //~^ ERROR: deny(unused) incompatible with previous forbid
    //~| WARNING being phased out
    #![warn(unused)]
    #![allow(unused)]

    num * num
}

fn forbid_last(num: i32) -> i32 {
    #![deny(unused)]
    #![warn(unused)]
    #![allow(unused)]
    #![forbid(unused)]

    num * num
}

fn forbid_multiple(num: i32) -> i32 {
    #![forbid(unused)]
    #![forbid(unused)]

    num * num
}

fn main() {
    forbid_first(10);
    forbid_last(10);
    forbid_multiple(10);
}
