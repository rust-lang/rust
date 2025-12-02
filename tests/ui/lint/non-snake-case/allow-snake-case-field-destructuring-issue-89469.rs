// Regression test for #89469, where an extra non_snake_case warning was
// reported for a shorthand field binding.
#![allow(todo_macro_calls)]

//@ check-pass
#![deny(non_snake_case)]

#[allow(non_snake_case)]
struct Entry {
    A: u16,
    a: u16
}

fn foo() -> Entry {todo!()}

pub fn f() {
    let Entry { A, a } = foo();
    let _ = (A, a);
}

fn main() {}
