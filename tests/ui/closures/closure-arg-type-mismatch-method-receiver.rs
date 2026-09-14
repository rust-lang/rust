// Regression test for https://github.com/rust-lang/rust/issues/156299.

struct A;

fn foo(_: &A) {}

fn test_foo() {
    (|a: A| foo(a)).bar();
        //~^ ERROR mismatched types
        //~| ERROR no method named `bar` found
}

fn main() {}
