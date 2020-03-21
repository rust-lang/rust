#![feature(negative_impls)]

pub trait ForeignTrait {}

impl ForeignTrait for u32 {}
impl !ForeignTrait for String {}
