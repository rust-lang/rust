#![feature(negative_impls)]
#![feature(with_negative_coherence)]

pub trait ForeignTrait {}

impl ForeignTrait for u32 {}
impl !ForeignTrait for String {}
