//! Ensure that we refuse to run a do_not_const_check function, even if the body *would* const-check
//! at the moment.

//@ rustc-env:RUSTC_ICE=0
//@ failure-status: 101
//@ normalize-stderr: "note: compiler flags.*\n\n" -> ""
//@ normalize-stderr: "note: rustc.*running on.*" -> "note: rustc {version} running on {platform}"
//@ normalize-stderr: "thread 'rustc'.*panicked.*:\n.*\n" -> ""
//@ normalize-stderr: " +\d{1,}: .*\n" -> ""
//@ normalize-stderr: " + at .*\n" -> ""
//@ normalize-stderr: " +.*omitted.*frames?.*\n" -> ""
//@ normalize-stderr: ".*note: Some details are omitted.*\n" -> ""
//@ normalize-stderr: "(internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "

#![feature(rustc_attrs, intrinsics)]

#[rustc_do_not_const_check]
const fn mostly_harmless() {}

const _: () = {
    mostly_harmless(); //~ERROR: calling non-const function
};

// Also ensure the same happens with intrinsics.
// Here we need some intrinsic that the interpreter does *not* have a native implementation for.
// Let's hope nobody adds one...
#[rustc_intrinsic]
#[rustc_do_not_const_check]
pub const fn integer_min<T: Copy>(a: T, b: T) -> T {
    a
}

const _: () = {
    integer_min(0, 1); //~ERROR: calling non-const function
};

fn main() {}
