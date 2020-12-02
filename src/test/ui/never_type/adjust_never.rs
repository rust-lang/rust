// Test that a variable of type ! can coerce to another type.

// check-pass

#![feature(never_type)]

fn main() {
    let x: ! = panic!();
    let y: u32 = x;
}

fn foo() {
    let x: never = panic!();
    let y: u32 = x;
}
