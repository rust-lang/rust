// compile-flags: -Zunleash-the-miri-inside-of-you

#![allow(const_err)]

// A test demonstrating that we prevent calling non-const fn during CTFE.

fn foo() {}

static C: () = foo(); //~ WARN: skipping const checks
//~^ ERROR could not evaluate static initializer
//~| NOTE calling non-const function `foo`

fn main() {}
