#![feature(doc_label_trait)]
#![crate_name = "foo"]

#[doc(label_trait)]
pub trait Labeled {}

pub trait Bound {}

// A conditional impl: the badge is rendered unconditionally even though the
// impl only holds for `T: Bound`.
//@ has 'foo/struct.Wrapper.html' '//a[@class="impl-label-trait-full-badge"]' 'Labeled'
pub struct Wrapper<T>(pub T);
impl<T: Bound> Labeled for Wrapper<T> {}
