//! This test checks that when there's a type mismatch between a function item and
//! a function pointer, the error message focuses on the actual type difference
//! (return types, argument types) rather than the confusing "pointer vs item" distinction.
//!
//! See https://github.com/rust-lang/rust/issues/127263

fn bar() {}

fn foo(x: i32) -> u32 {
    0
}

fn main() {
    let b: fn() -> u32 = bar; //~ ERROR mismatched types [E0308]
    let f: fn(i32) = foo; //~ ERROR mismatched types [E0308]
}
