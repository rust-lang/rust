#![feature(no_core, auto_traits, lang_items, arbitrary_self_types)]
#![no_core]

#[lang = "sized"]
trait Sized {}

pub auto trait Bar {}

/// has span
impl Foo {
    pub fn baz(&self) {}
}

// Testing spans, so all tests below code
//@ is "$.index[*][?(@.docs=='has span')].span.begin" "[10, 0]"
//@ is "$.index[*][?(@.docs=='has span')].span.end" "[12, 1]"
// FIXME: this doesn't work due to https://github.com/freestrings/jsonpath/issues/91
// is "$.index[*][?(@.inner.impl.is_synthetic==true)].span" null
pub struct Foo;
