#![feature(doc_notable_trait, negative_impls)]

#[doc(notable_trait)]
pub trait SomeTrait {}

pub struct Positive;
impl SomeTrait for Positive {}

pub struct Negative;
impl !SomeTrait for Negative {}

//@ has doc_notable_trait_negative/fn.positive.html
//@ snapshot positive - '//script[@id="notable-traits-data"]'
pub fn positive() -> Positive {
    todo!()
}

//@ has doc_notable_trait_negative/fn.negative.html
//@ count - '//script[@id="notable-traits-data"]' 0
pub fn negative() -> Negative {
    &[]
}
