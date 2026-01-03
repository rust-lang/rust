//@ edition:2015..2021
trait X {
  type Y<'a>;
}

fn foo<'a>(arg: Box<dyn X<Y('a) = &'a ()>>) {}
  //~^ ERROR: lifetimes must be followed by `+` to form a trait object type
  //~| ERROR: parenthesized generic arguments cannot be used
  //~| ERROR associated type takes 0 generic arguments but 1 generic argument
  //~| ERROR associated type takes 1 lifetime argument but 0 lifetime arguments


fn bar<'a>(arg: Box<dyn X<Y() = ()>>) {}
  //~^ ERROR: parenthesized generic arguments cannot be used
  //~| ERROR associated type takes 1 lifetime argument but 0 lifetime arguments

fn main() {}
