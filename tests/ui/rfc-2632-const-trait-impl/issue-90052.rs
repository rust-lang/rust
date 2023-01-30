#![feature(const_trait_impl)]

#[const_trait]
trait Bar {}

fn foo<T>() where T: ~const Bar {}
//~^ ERROR `~const` is not allowed

fn main() {}
