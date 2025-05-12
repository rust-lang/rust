#![crate_name = "foo"]

//@ has 'foo/index.html'
// There should be only `type A`.
//@ count - '//*[@class="item-table"]//dt' 1
//@ has - '//dt/a[@href="type.A.html"]' 'A'

mod foo {
    pub struct S;
}

use foo::S;

pub type A = S;

//@ has 'foo/type.A.html'
//@ has - '//*[@id="method.default"]/h4' 'fn default() -> Self'
impl Default for A {
    fn default() -> Self {
        S
    }
}

//@ has - '//*[@id="method.a"]/h4' 'pub fn a(&self)'
impl A {
    pub fn a(&self) {}
}
