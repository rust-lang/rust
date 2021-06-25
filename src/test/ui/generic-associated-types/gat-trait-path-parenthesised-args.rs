#![feature(generic_associated_types)]
  //~^ WARNING: the feature `generic_associated_types` is incomplete

trait X {
  type Y<'a>;
}

fn foo<'a>(arg: Box<dyn X<Y('a) = &'a ()>>) {}
  //~^ ERROR: lifetime in trait object type must be followed by `+`
  //~| ERROR: parenthesized generic arguments cannot be used
  //~| ERROR this associated type takes 0 generic arguments but 1 generic argument
  //~| ERROR this associated type takes 1 lifetime argument but 0 lifetime arguments
  //~| WARNING: trait objects without an explicit `dyn` are deprecated
  //~| WARNING: this is accepted in the current edition

fn main() {}
