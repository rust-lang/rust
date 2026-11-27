// Regression test for <https://github.com/rust-lang/rust/issues/159264>

#![feature(checked_type_aliases, pin_ergonomics)]
#![allow(incomplete_features)]

#[pin_v2]
pub struct UnpinLocal<T> {
    pub foo: T
}

pub type UnpinLocalAlias<T> = UnpinLocal<T>;
impl <T> Unpin for UnpinLocalAlias<T> {}
//~^ ERROR: explicit impls for the `Unpin` trait are not permitted for structurally pinned types

fn main() {}
