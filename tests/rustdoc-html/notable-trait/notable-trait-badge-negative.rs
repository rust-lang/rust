#![feature(doc_notable_trait, negative_impls)]
#![crate_name = "foo"]

#[doc(notable_trait)]
pub trait Labeled {}

// A negative impl must not produce a badge.
//@ count 'foo/struct.Neg.html' '//div[@class="notable-trait-badge-container"]' 0
pub struct Neg;
impl !Labeled for Neg {}
