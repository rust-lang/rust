#![warn(clippy::rest_pat_in_fully_bound_structs)]

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
        A { a: 0, b: 0, c: "", .. } => {},  // Lint
        _ => {},
    }

    match a_struct {
        A { a: 5, b: 42, .. } => {},
        A { a: 0, b: 0, c: "", .. } => {}, // Lint
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
}
