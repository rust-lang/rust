// compile-flags: -Z print-type-sizes --crate-type lib
// build-pass
// ignore-pass

// Tests a generator that has its discriminant as the *final* field.

// Avoid emitting panic handlers, like the rest of these tests...
#![feature(generators)]
#![allow(drop_copy)]

pub fn foo() {
    let a = || {
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
