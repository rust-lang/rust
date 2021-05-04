#![feature(generic_associated_types)]
  //~^ WARNING: the feature `generic_associated_types` is incomplete

trait X {
  type Y<'a>;
    //~^ ERROR this associated type
    //~| ERROR this associated type
}

fn foo<'a>(arg: Box<dyn X<Y('a) = &'a ()>>) {}
  //~^ ERROR: lifetime in trait object type must be followed by `+`
  //~| ERROR: parenthesized generic arguments cannot be used
  //~| WARNING: trait objects without an explicit `dyn` are deprecated
  //~| WARNING: this was previously accepted by the compiler

fn main() {}
