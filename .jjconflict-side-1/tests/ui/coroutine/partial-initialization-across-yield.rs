// Test that we don't allow yielding from a coroutine while a local is partially
// initialized.

#![feature(coroutines, stmt_expr_attributes)]

struct S { x: i32, y: i32 }
struct T(i32, i32);

fn test_tuple() {
    let _ = #[coroutine] || {
        let mut t: (i32, i32);
        t.0 = 42; //~ ERROR E0381
        yield;
        t.1 = 88;
        let _ = t;
    };
}

fn test_tuple_struct() {
    let _ = #[coroutine] || {
        let mut t: T;
        t.0 = 42; //~ ERROR E0381
        yield;
        t.1 = 88;
        let _ = t;
    };
}

fn test_struct() {
    let _ = #[coroutine] || {
        let mut t: S;
        t.x = 42; //~ ERROR E0381
        yield;
        t.y = 88;
        let _ = t;
    };
}

fn main() {
    test_tuple();
    test_tuple_struct();
    test_struct();
}
