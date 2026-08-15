#![feature(doc_notable_trait, negative_impls)]
#![crate_name = "foo"]

#[doc(notable_trait)]
pub trait Neg {}
#[doc(notable_trait)]
pub trait Pos {}

// A negative impl must not produce a badge.
//@ has 'foo/struct.T.html'
//@ count - '//div[@class="notable-trait-badge-container"]/a' 1
//@ has - '//div[@class="notable-trait-badge-container"]/a[@href="trait.Pos.html"]' 'Pos'
pub struct T;
impl !Neg for T {}
impl Pos for T {}
