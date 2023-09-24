trait A {}

impl A .. {}
//~^ ERROR missing `for` in a trait impl
//~| ERROR `impl Trait for .. {}` is an obsolete syntax for auto traits

impl A      usize {}
//~^ ERROR missing `for` in a trait impl

fn main() {}
