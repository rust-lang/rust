//@ known-bug: #121161
#![allow(incomplete_features)]
#![feature(unnamed_fields)]


#[derive(Eq)]
#[repr(C)]
struct Bar {
    _: union {
        a: u8,
    },
}
