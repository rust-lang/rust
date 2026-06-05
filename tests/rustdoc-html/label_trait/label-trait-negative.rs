#![feature(doc_label_trait, negative_impls)]
#![crate_name = "foo"]

#[doc(label_trait)]
pub trait Labeled {}

// A negative impl must not produce a badge.
//@ count 'foo/struct.Neg.html' '//div[@class="impl-label-trait-full-badge-container"]' 0
pub struct Neg;
impl !Labeled for Neg {}
