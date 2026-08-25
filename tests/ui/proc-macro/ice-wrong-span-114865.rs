// Regression test for ICE https://github.com/rust-lang/rust/issues/114865

// Tests that we do not ICE when a proc macro expands
// to a string formatting macro (like println!) and respans
// this formatting macro's arg to that of its own input which
// happens to be a multi-byte string (see the auxiliary file
// ice-wrong-span-114865.rs).

//@ proc-macro: ice-wrong-span-114865.rs

extern crate ice_wrong_span_114865;

use ice_wrong_span_114865::{foo, foo2, foo3, foo4};

fn main() {
    foo!("字"); //~ ERROR 1 positional argument in format string, but no arguments were given
    foo!("r字字"); //~ ERROR 1 positional argument in format string, but no arguments were given

    foo2!("字"); //~ ERROR 1 positional argument in format string, but no arguments were given
    foo2!("r字字"); //~ ERROR 1 positional argument in format string, but no arguments were given

    foo3!(r"abc" 字); //~ ERROR 1 positional argument in format string, but no arguments were given

    foo4!(r##"abcd"## 字); //~ ERROR 1 positional argument in format string, but no arguments were given
}
