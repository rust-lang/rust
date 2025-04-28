#![feature(no_core, auto_traits, lang_items, arbitrary_self_types)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[lang = "legacy_receiver"]
pub trait LegacyReceiver {}

pub auto trait Bar {}

/// has span
impl Foo {
    pub fn baz(&self) {}
}

// Testing spans, so all tests below code
//@ is "$.index[?(@.docs=='has span')].span.begin" "[13, 1]"
//@ is "$.index[?(@.docs=='has span')].span.end" "[15, 2]"
//@ is "$.index[?(@.docs=='has span')].inner.impl.is_synthetic" false
//@ is "$.index[?(@.inner.impl.is_synthetic==true)].span" null
//@ is "$.index[?(@.inner.impl.is_synthetic==true)].inner.impl.for.resolved_path.path" '"Foo"'
//@ is "$.index[?(@.inner.impl.is_synthetic==true)].inner.impl.trait.path" '"Bar"'
pub struct Foo;
