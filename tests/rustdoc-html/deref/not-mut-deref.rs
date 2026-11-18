// This test ensures that `get_u32` is not listed as the `Deref` impl
// doesn't match it.
// Regression test for <https://github.com/rust-lang/rust/issues/24686>.

#![crate_name = "foo"]

//@ has 'foo/struct.Bar.html'
//@ count - '//*[@id="deref-methods-Foo%3Ci32%3E-1"]/*[@class="method"]' 1
//@ has - '//*[@id="deref-methods-Foo%3Ci32%3E-1"]/*[@class="method"]/h4' \
// 'pub fn get_i32(&self) -> i32'

// Same check for the sidebar.
//@ count - '//*[@id="rustdoc-toc"]/*[@class="block deref-methods"]//a' 1
//@ has - '//*[@id="rustdoc-toc"]/*[@class="block deref-methods"]//a' 'get_i32'

use std::ops::Deref;

pub struct Foo<T>(T);

impl Foo<i32> {
    pub fn get_i32(&self) -> i32 { self.0 }
}

impl Foo<u32> {
    pub fn get_u32(&self) -> u32 { self.0 }
}

pub struct Bar(Foo<i32>);

impl Deref for Bar {
    type Target = Foo<i32>;
    fn deref(&self) -> &Foo<i32> {
        &self.0
    }
}
