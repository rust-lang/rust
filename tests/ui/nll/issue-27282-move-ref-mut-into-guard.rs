// Issue 27282: Example 1: This sidesteps the AST checks disallowing
// mutable borrows in match guards by hiding the mutable borrow in a
// guard behind a move (of the ref mut pattern id) within a closure.

#![feature(if_let_guard)]

fn main() {
    match Some(&4) {
        None => {},
        ref mut foo
            if { (|| { let mut bar = foo; bar.take() })(); false } => {},
        //~^ ERROR cannot move out of `foo` in pattern guard [E0507]
        Some(s) => std::process::exit(*s),
    }

    match Some(&4) {
        None => {},
        ref mut foo
            if let Some(()) = { (|| { let mut bar = foo; bar.take() })(); None } => {},
        //~^ ERROR cannot move out of `foo` in pattern guard [E0507]
        Some(s) => std::process::exit(*s),
    }
}
