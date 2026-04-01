#![warn(clippy::rest_pat_in_fully_bound_structs)]
#![allow(clippy::struct_field_names)]

struct A {
    a: i32,
    b: i64,
    c: &'static str,
}

macro_rules! foo {
    ($param:expr) => {
        match $param {
            A { a: 0, b: 0, c: "", .. } => {},
            _ => {},
        }
    };
}

fn main() {
    let a_struct = A { a: 5, b: 42, c: "A" };

    match a_struct {
        A { a: 5, b: 42, c: "", .. } => {}, // Lint
        //~^ rest_pat_in_fully_bound_structs
        A { a: 0, b: 0, c: "", .. } => {}, // Lint
        //~^ rest_pat_in_fully_bound_structs
        _ => {},
    }

    match a_struct {
        A { a: 5, b: 42, .. } => {},
        A { a: 0, b: 0, c: "", .. } => {}, // Lint
        //~^ rest_pat_in_fully_bound_structs
        _ => {},
    }

    // No lint
    match a_struct {
        A { a: 5, .. } => {},
        A { a: 0, b: 0, .. } => {},
        _ => {},
    }

    // No lint
    foo!(a_struct);

    #[non_exhaustive]
    struct B {
        a: u32,
        b: u32,
        c: u64,
    }

    let b_struct = B { a: 5, b: 42, c: 342 };

    match b_struct {
        B { a: 5, b: 42, .. } => {},
        B { a: 0, b: 0, c: 128, .. } => {}, // No Lint
        _ => {},
    }
}
