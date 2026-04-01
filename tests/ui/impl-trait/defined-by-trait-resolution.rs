//! The trait query `foo: Fn() -> u8` is a valid defining use of RPIT.

//@ build-pass

fn returns_u8(_: impl Fn() -> u8) {}

pub fn foo() -> impl Sized {
    returns_u8(foo);
    0u8
}

fn main() {}
