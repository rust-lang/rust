fn foo<const SIZE: usize = 5>() {}
//~^ ERROR default values for const generic parameters are experimental

fn main() {}
