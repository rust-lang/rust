#![allow(warnings)]

// This tests that the `help: consider dereferencing the boxed value` suggestion isn't made
// because the box doesn't deref to the type of the arm.

enum S {
    A { a: usize },
    B { b: usize },
}

fn main() {
    let x = Box::new(3u32);
    let y = match x {
        S::A { a } | S::B { b: a } => a,
        //~^ ERROR mismatched types [E0308]
        //~^^ ERROR mismatched types [E0308]
    };
}
