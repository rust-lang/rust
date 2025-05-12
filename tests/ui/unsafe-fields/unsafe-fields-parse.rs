//@ compile-flags: --crate-type=lib
#![allow(incomplete_features)]
#![feature(unsafe_fields)]

// Parse errors even *with* unsafe_fields, which would make the compiler early-exit otherwise.
enum A {
    TupleLike(unsafe u32), //~ ERROR
}

struct B(unsafe u32); //~ ERROR
