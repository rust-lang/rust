// This test ensures that methods on type aliases generate a URL to themselves matching
// their ID attribute.
// Regression test for <https://github.com/rust-lang/rust/issues/131028>.

#![crate_name = "foo"]

pub struct X<T>(T);

//@ has 'foo/type.Y.html'
//@ has - '//section[@id="method.bar"]/h4' 'pub fn bar()'
//@ has - '//section[@id="method.bar"]/h4/a[@href="#method.bar"]' 'bar'
pub type Y = X<u32>;

impl Y {
    pub fn bar() {}
}
