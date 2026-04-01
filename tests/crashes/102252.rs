//@ known-bug: #102252

#![feature(min_specialization, rustc_attrs)]

#[rustc_specialization_trait]
pub trait Trait {}

struct Struct
where
    Self: Iterator<Item = <Self as Iterator>::Item>, {}

impl Trait for Struct {}

fn main() {}
