trait A {}

impl A .. {}
//~^ ERROR missing `for` in a trait impl
//~| ERROR `impl Trait for .. {}` is an obsolete syntax

impl A      usize {}
//~^ ERROR missing `for` in a trait impl

fn main() {}
