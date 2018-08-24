// Test use of const let without feature gate.

#![feature(const_fn)]

const fn foo() -> usize {
    let x = 42;
    //~^ ERROR statements in constant functions are unstable
    //~| ERROR: let bindings in constant functions are unstable
    42
}

fn main() {}
