// compile-flags: -Z print-type-sizes
// build-pass
// ignore-pass

// Tests a generator that has its discriminant as the *final* field.

// Avoid emitting panic handlers, like the rest of these tests...
#![feature(start, generators)]

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
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
    0
}
