// Test that we don't allow yielding from a generator while a local is partially
// initialized.

#![feature(generators)]

struct S { x: i32, y: i32 }
struct T(i32, i32);

fn test_tuple() {
    let _ = || {
        let mut t: (i32, i32);
        t.0 = 42;
        //~^ ERROR assign to part of possibly-uninitialized variable: `t` [E0381]
        yield;
        t.1 = 88;
        let _ = t;
    };
}

fn test_tuple_struct() {
    let _ = || {
        let mut t: T;
        t.0 = 42;
        //~^ ERROR assign to part of possibly-uninitialized variable: `t` [E0381]
        yield;
        t.1 = 88;
        let _ = t;
    };
}

fn test_struct() {
    let _ = || {
        let mut t: S;
        t.x = 42;
        //~^ ERROR assign to part of possibly-uninitialized variable: `t` [E0381]
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
