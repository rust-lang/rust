#![feature(no_core, auto_traits, lang_items, arbitrary_self_types)]
#![feature(const_trait_impl)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
#[const_trait]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
#[const_trait]
pub trait Sized: MetaSized {}

#[lang = "legacy_receiver"]
pub trait LegacyReceiver {}

pub auto trait Bar {}

/// has span
impl Foo {
    pub fn baz(&self) {}
}

// Testing spans, so all tests below code
//@ is "$.index[?(@.docs=='has span')].span.begin" "[22, 0]"
//@ is "$.index[?(@.docs=='has span')].span.end" "[24, 1]"
// FIXME: this doesn't work due to https://github.com/freestrings/jsonpath/issues/91
// is "$.index[?(@.inner.impl.is_synthetic==true)].span" null
pub struct Foo;
