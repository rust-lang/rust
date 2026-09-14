#![feature(doc_notable_trait)]
#![crate_name = "foo"]

#[doc(notable_trait)]
pub trait Base {}

pub trait Derived: Base {}

//@ has 'foo/struct.S.html'
// Implementing `Derived` requires implementing the notable supertrait `Base`,
// so its badge shows up.
//@ count - '//div[@class="notable-trait-badge-container"]/a' 1
//@ has - '//div[@class="notable-trait-badge-container"]/a[@href="trait.Base.html"]' 'Base'
pub struct S;
impl Base for S {}
impl Derived for S {}
