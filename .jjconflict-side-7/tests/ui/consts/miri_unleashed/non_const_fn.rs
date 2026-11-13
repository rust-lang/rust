//@ compile-flags: -Zunleash-the-miri-inside-of-you

// A test demonstrating that we prevent calling non-const fn during CTFE.

fn foo() {}

static C: () = foo();
//~^ ERROR calling non-const function `foo`

fn main() {}

//~? WARN skipping const checks
