// Ensures that aliased types also get the "methods from deref" content.
// Regression test for <https://github.com/rust-lang/rust/issues/134868>.

#![crate_name = "foo"]

use std::ops::Deref;

// First we check for the `Col` struct.
//@ has 'foo/struct.Col.html'
//@ has - '//*[@id="main-content"]//h2[@id="deref-methods-%5BT%5D"]' 'Methods from Deref<Target = [T]>'
//@ has - '//*[@id="rustdoc-toc"]//a[@href="#deref-methods-%5BT%5D"]' 'Methods from Deref<Target=[T]>'

pub struct Col<T: Copy, const N: usize> {
    data: [T; N],
}

// Then for the type alias.
//@ has 'foo/type.Mat.html'
//@ has - '//*[@id="main-content"]//h2[@id="deref-methods-%5BT%5D"]' 'Methods from Deref<Target = [T]>'
//@ has - '//*[@id="rustdoc-toc"]//a[@href="#deref-methods-%5BT%5D"]' 'Methods from Deref<Target=[T]>'

pub type Mat = Col<u8, 32>;

impl<T: Copy, const N: usize> Deref for Col<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
