// Checks that `: Self` never gets emitted for simple `self` parameters.

#![crate_name = "foo"]

pub struct Foo;

// @!has foo/struct.Foo.html '//*[@class="impl-items"]' Self
impl Foo {
    pub fn by_value(self) {}
    pub fn by_value_mut(mut self) {}
    pub fn by_ref(&self) {}
    pub fn by_mut_ref(&mut self) {}
    pub fn by_value_explicit(self: Self) {}
    pub fn by_value_mut_explicit(mut self: Self) {}
}
