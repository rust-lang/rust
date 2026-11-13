#![feature(doc_notable_trait)]
#![crate_name = "foo"]

#[doc(notable_trait)]
pub trait Labeled {}

pub trait Bound {}

// A conditional impl: the badge is rendered unconditionally even though the
// impl only holds for `T: Bound`.
//@ has 'foo/struct.Wrapper.html' '//div[@class="notable-trait-badge-container"]/a[@href="trait.Labeled.html"]' 'Labeled'
pub struct Wrapper<T>(pub T);
impl<T: Bound> Labeled for Wrapper<T> {}
