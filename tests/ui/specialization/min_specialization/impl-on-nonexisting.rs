#![feature(min_specialization)]

trait Trait {}
impl Trait for NonExistent {}
//~^ ERROR cannot find type `NonExistent`

fn main() {}
