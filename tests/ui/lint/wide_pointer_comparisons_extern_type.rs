// Pointers to `extern type`s (and to structs whose tail is one) are thin, so
// comparing them is not an "ambiguous wide pointer comparison".
//@ check-pass

#![feature(extern_types)]
#![deny(ambiguous_wide_pointer_comparisons)]
#![allow(dead_code)]

use std::ptr::NonNull;

unsafe extern "C" {
    type Cat;
}

struct Wrapper {
    _x: u8,
    tail: Cat,
}

fn eq(a: *const Cat, b: *const Cat) -> bool {
    a == b
}

fn ne(a: *mut Cat, b: *mut Cat) -> bool {
    a != b
}

fn method(a: *const Cat, b: *const Cat) -> bool {
    a.eq(&b)
}

fn wrapper(a: *const Wrapper, b: *const Wrapper) -> bool {
    a == b
}

fn nonnull(a: NonNull<Cat>, b: NonNull<Cat>) -> bool {
    a == b
}

fn main() {}
