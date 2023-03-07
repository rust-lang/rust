#![feature(unsized_locals)]
//~^ WARN the feature `unsized_locals` is incomplete

struct Test([i32]);

fn main() {
    let _x: fn(_) -> Test = Test;
} //~^the size for values of type `[i32]` cannot be known at compilation time
