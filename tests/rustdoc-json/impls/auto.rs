#![feature(no_core, auto_traits, lang_items, arbitrary_self_types)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[lang = "legacy_receiver"]
pub trait LegacyReceiver {}

pub auto trait Bar {}

/// has span
impl Foo {
    pub fn baz(&self) {}
}

// Testing spans, so all tests below code
//@ is "$.index[?(@.docs=='has span')].span.begin" "[19, 1]"
//@ is "$.index[?(@.docs=='has span')].span.end" "[21, 2]"
//@ is "$.index[?(@.docs=='has span')].inner.impl.is_synthetic" false
//@ is "$.index[?(@.inner.impl.is_synthetic==true)].span" null
//@ is "$.index[?(@.inner.impl.is_synthetic==true)].inner.impl.for.resolved_path.path" '"Foo"'
//@ is "$.index[?(@.inner.impl.is_synthetic==true)].inner.impl.trait.path" '"Bar"'
pub struct Foo;
