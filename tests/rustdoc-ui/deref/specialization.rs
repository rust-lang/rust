// This test ensures that specializing the `Deref` trait won't panic.
// Regression test for <https://github.com/rust-lang/rust/issues/156484>.

//@ check-pass

#![feature(specialization)]

use std::ops::Deref;

pub struct Foo<T> {
    data_pd: T,
}

impl<T> Deref for Foo<T> {
    type Target = i32;
    fn deref(&self) -> &Self::Target {
        self
    }
}

pub struct TagA;
impl Deref for Foo<TagA> {}
