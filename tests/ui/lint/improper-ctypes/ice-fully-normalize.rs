//@ check-pass
#![deny(improper_ctypes)]

// Issue: https://github.com/rust-lang/rust/issues/73249
// "ICE: could not fully normalize"

pub trait Foo {
    type Assoc;
}

impl Foo for () {
    type Assoc = u32;
}

extern "C" {
    pub fn lint_me(x: Bar<()>);
}

#[repr(transparent)]
pub struct Bar<T: Foo> {
    value: <T as Foo>::Assoc,
}

fn main() {}
