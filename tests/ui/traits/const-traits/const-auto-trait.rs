#![feature(auto_traits, const_trait_impl)]

const auto trait Marker {}
//~^ ERROR: auto traits cannot be const

fn main() {}
