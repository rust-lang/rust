//@ run-pass
//@ proc-macro: test-macros.rs
//@ compile-flags: -Z span-debug
//@ edition:2018
//
// Tests the pretty-printing behavior of inserting `Invisible`-delimited groups

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

extern crate test_macros;
use test_macros::print_bang_consume;

macro_rules! expand_it {
    (($val1:expr) ($val2:expr)) => { expand_it!($val1 + $val2) };
    ($val:expr) => { print_bang_consume!("hi" $val (1 + 1)) };
}

fn main() {
    expand_it!(1 + (25) + 1);
    expand_it!(("hello".len()) ("world".len()));
    f();
}

// The key thing here is to produce a single `None`-delimited `Group`, even
// though there is multiple levels of macros.
macro_rules! m5 { ($e:expr) => { print_bang_consume!($e) }; }
macro_rules! m4 { ($e:expr) => { m5!($e); } }
macro_rules! m3 { ($e:expr) => { m4!($e); } }
macro_rules! m2 { ($e:expr) => { m3!($e); } }
macro_rules! m1 { ($e:expr) => { m2!($e); } }

fn f() {
    m1!(123);
}
