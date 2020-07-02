// run-pass
// aux-build:test-macros.rs
// compile-flags: -Z span-debug
// edition:2018
//
// Tests the pretty-printing behavior of inserting `NoDelim` groups

extern crate test_macros;
use test_macros::print_bang_consume;

macro_rules! expand_it {
    (($val1:expr) ($val2:expr)) => { expand_it!($val1 + $val2) };
    ($val:expr) => { print_bang_consume!("hi" $val (1 + 1)) };
}

fn main() {
    expand_it!(1 + (25) + 1);
    expand_it!(("hello".len()) ("world".len()));
}
