//@ check-pass
#![warn(unused_parens)]

fn foo() {
    return (_ = 42);
    // lint unused_parens should not be triggered here.
}

fn main() {}
