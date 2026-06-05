#![feature(doc_label_trait)]
#![crate_name = "foo"]

#[doc(label_trait)]
pub trait Base {}

pub trait Derived: Base {}

//@ has 'foo/struct.S.html'
// Implementing `Derived` requires implementing the label supertrait `Base`,
// so its badge shows up.
//@ count - '//a[@class="impl-label-trait-full-badge"]' 1
//@ has - '//a[@class="impl-label-trait-full-badge"]' 'Base'
pub struct S;
impl Base for S {}
impl Derived for S {}
