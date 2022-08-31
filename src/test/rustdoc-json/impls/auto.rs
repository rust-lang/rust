#![feature(no_core, auto_traits, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

pub auto trait Bar {}

/// has span
impl Foo {
    pub fn baz(&self) {}
}

// Testing spans, so all tests below code
// @is "$.index[*][?(@.kind=='impl' && @.inner.synthetic==true)].span" null
// @is "$.index[*][?(@.docs=='has span')].span.begin" "[10, 0]"
// @is "$.index[*][?(@.docs=='has span')].span.end" "[12, 1]"
pub struct Foo;
