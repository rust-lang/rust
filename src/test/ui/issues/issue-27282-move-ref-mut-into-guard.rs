// Issue 27282: Example 1: This sidesteps the AST checks disallowing
// mutable borrows in match guards by hiding the mutable borrow in a
// guard behind a move (of the ref mut pattern id) within a closure.
//
// This example is not rejected by AST borrowck (and then reliably
// segfaults when executed).

#![feature(nll)]

fn main() {
    match Some(&4) {
        None => {},
        ref mut foo
            if { (|| { let bar = foo; bar.take() })(); false } => {},
        //~^ ERROR cannot move out of `foo` in pattern guard [E0507]
        Some(s) => std::process::exit(*s),
    }
}
