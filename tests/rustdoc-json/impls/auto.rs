#![feature(no_core, rustc_attrs, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[rustc_auto_trait]
pub trait Bar {}

/// has span
impl Foo {
    pub fn baz(&self) {}
}

// Testing spans, so all tests below code
// @is "$.index[*][?(@.docs=='has span')].span.begin" "[11, 0]"
// @is "$.index[*][?(@.docs=='has span')].span.end" "[13, 1]"
// FIXME: this doesn't work due to https://github.com/freestrings/jsonpath/issues/91
// is "$.index[*][?(@.inner.impl.synthetic==true)].span" null
pub struct Foo;
