//@ proc-macro: proc_macro_sequence.rs
//@ ignore-backends: gcc

// Regression test for issue #62831: Check that multiple sequences with the same span in the
// left-hand side of a macro definition behave as if they had unique spans, and in particular that
// they don't crash the compiler.

#![allow(unused_macros)]

extern crate proc_macro_sequence;

// When ignoring spans, this macro has the same macro definition as `generated_foo` in
// `proc_macro_sequence.rs`.
macro_rules! manual_foo {
    (1 $x:expr $($y:tt,)*   //~ERROR `$x:expr` may be followed by `$y:tt`
               $(= $z:tt)*  //~ERROR `$x:expr` may be followed by `=`
    ) => {};
}

proc_macro_sequence::make_foo!(); //~ERROR `$x:expr` may be followed by `$y:tt`
                                  //~^ERROR `$x:expr` may be followed by `=`

fn main() {}
