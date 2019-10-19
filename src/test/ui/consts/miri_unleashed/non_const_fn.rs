// compile-flags: -Zunleash-the-miri-inside-of-you
#![allow(const_err)]

// A test demonstrating that we prevent calling non-const fn during CTFE.

fn foo() {}

const C: () = foo(); //~ WARN: skipping const checks

fn main() {
    println!("{:?}", C); //~ ERROR: evaluation of constant expression failed
}
