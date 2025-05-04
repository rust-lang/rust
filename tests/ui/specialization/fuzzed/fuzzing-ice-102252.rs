// This test previous triggered an assertion that there are no inference variables
// returned by `wf::obligations`. Overflow when normalizing
// `Self: Iterator<Item = <Self as Iterator>::Item>` resulted in overflow which then
// caused us to return an infer var.
//
// This assert has now been removed.
#![feature(min_specialization, rustc_attrs)]

#[rustc_specialization_trait]
pub trait Trait {}

struct Struct
//~^ ERROR overflow evaluating the requirement `<Struct as Iterator>::Item == _`
where
    Self: Iterator<Item = <Self as Iterator>::Item>, {}

impl Trait for Struct {}
//~^ ERROR `Struct` is not an iterator

fn main() {}
