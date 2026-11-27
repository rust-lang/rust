#![feature(doc_notable_trait, negative_impls)]

#[doc(notable_trait)]
pub trait SomeTrait {}

pub struct Positive;
impl SomeTrait for Positive {}

pub struct Negative;
impl !SomeTrait for Negative {}

//@ has doc_notable_trait_negative/fn.positive.notable-traits-data.js
//@ snapshot positive -
pub fn positive() -> Positive {
    todo!()
}

//@ has doc_notable_trait_negative/fn.negative.html
//@ !has doc_notable_trait_negative/fn.negative.notable-traits-data.js
pub fn negative() -> Negative {
    &[]
}
