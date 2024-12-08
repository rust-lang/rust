//@ run-pass
//@ needs-unwind
#![allow(unused_imports)]
// Ideally, any macro call with a trailing comma should behave
// identically to a call without the comma.
//
// This checks the behavior of macros with trailing commas in key
// places where regressions in behavior seem highly possible (due
// to it being e.g., a place where the addition of an argument
// causes it to go down a code path with subtly different behavior).
//
// There is a companion failing test.

//@ compile-flags: --test -C debug_assertions=yes
//@ revisions: std core

#![cfg_attr(core, no_std)]

#[cfg(core)]
use core::fmt;
#[cfg(std)]
use std::fmt;

// an easy mistake in the implementation of 'assert!'
// would cause this to say "explicit panic"
#[test]
#[should_panic(expected = "assertion failed")]
fn assert_1arg() {
    assert!(false,);
}

// same as 'assert_1arg'
#[test]
#[should_panic(expected = "assertion failed")]
fn debug_assert_1arg() {
    debug_assert!(false,);
}

// make sure we don't accidentally forward to `write!("text")`
#[cfg(std)]
#[test]
fn writeln_1arg() {
    use fmt::Write;

    let mut s = String::new();
    writeln!(&mut s,).unwrap();
    assert_eq!(&s, "\n");
}

// A number of format_args-like macros have special-case treatment
// for a single message string, which is not formatted.
//
// This test ensures that the addition of a trailing comma does not
// suddenly cause these strings to get formatted when they otherwise
// would not be. This is an easy mistake to make by having such a macro
// accept ", $($tok:tt)*" instead of ", $($tok:tt)+" after its minimal
// set of arguments.
//
// (Example: Issue #48042)
#[test]
#[allow(non_fmt_panics)]
fn to_format_or_not_to_format() {
    // ("{}" is the easiest string to test because if this gets
    // sent to format_args!, it'll simply fail to compile.
    // "{{}}" is an example of an input that could compile and
    // produce an incorrect program, but testing the panics
    // would be burdensome.)
    let falsum = || false;

    assert!(true, "{}",);

    // assert_eq!(1, 1, "{}",); // see check-fail
    // assert_ne!(1, 2, "{}",); // see check-fail

    debug_assert!(true, "{}",);

    // debug_assert_eq!(1, 1, "{}",); // see check-fail
    // debug_assert_ne!(1, 2, "{}",); // see check-fail
    // eprint!("{}",); // see check-fail
    // eprintln!("{}",); // see check-fail
    // format!("{}",); // see check-fail
    // format_args!("{}",); // see check-fail

    if falsum() {
        panic!("{}",);
    }

    // print!("{}",); // see check-fail
    // println!("{}",); // see check-fail
    // unimplemented!("{}",); // see check-fail

    if falsum() {
        unreachable!("{}",);
    }

    // write!(&mut stdout, "{}",); // see check-fail
    // writeln!(&mut stdout, "{}",); // see check-fail
}
