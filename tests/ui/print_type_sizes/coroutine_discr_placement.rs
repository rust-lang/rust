//@ compile-flags: -Z print-type-sizes --crate-type lib
//@ build-pass
//@ ignore-pass

// Tests a coroutine that has its discriminant as the *final* field.

// Avoid emitting panic handlers, like the rest of these tests...
#![feature(coroutines, stmt_expr_attributes)]
#![allow(dropping_copy_types)]

pub fn foo() {
    let a = #[coroutine]
    || {
        {
            let w: i32 = 4;
            yield;
            drop(w);
        }
        {
            let z: i32 = 7;
            yield;
            drop(z);
        }
    };
}
