//! Regression test for <https://github.com/rust-lang/rust/issues/43806>.
//! Tests function calls which return `()`, `!`, and user defined
//! uninhabited types don't get false unused result lint.
//@ check-pass

#![deny(unused_results)]

enum Void {}

fn foo() {}

fn bar() -> ! {
    loop {}
}

fn baz() -> Void {
    loop {}
}

fn qux() {
    foo();
    bar();
    baz();
}

fn main() {}
