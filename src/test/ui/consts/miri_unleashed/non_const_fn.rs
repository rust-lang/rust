// build-fail
// compile-flags: -Zunleash-the-miri-inside-of-you

#![warn(const_err)]

// A test demonstrating that we prevent calling non-const fn during CTFE.

fn foo() {}

const C: () = foo(); //~ WARN: skipping const checks
//~^ WARN any use of this value will cause an error

fn main() {
    println!("{:?}", C);
    //~^ ERROR: evaluation of constant expression failed
    //~| WARN: erroneous constant used [const_err]
}
