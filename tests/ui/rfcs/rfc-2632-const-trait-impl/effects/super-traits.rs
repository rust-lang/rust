// check-pass

// misc tests not directly related to const traits, but encountered
// while implementing effects

#![feature(const_trait_impl, effects)]

// #[const_trait]
pub trait X<Rhs: ?Sized = Self> {}

pub trait Y: X {}

impl X for () {}

impl Y for () {}

fn main() {}
