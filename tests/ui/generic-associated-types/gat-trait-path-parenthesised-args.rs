trait X {
  type Y<'a>;
}

fn foo<'a>(arg: Box<dyn X<Y('a) = &'a ()>>) {}
  //~^ ERROR: lifetime in trait object type must be followed by `+`
  //~| ERROR: parenthesized generic arguments cannot be used
  //~| ERROR associated type takes 0 generic arguments but 1 generic argument
  //~| ERROR associated type takes 1 lifetime argument but 0 lifetime arguments
  //~| ERROR associated type takes 0 generic arguments but 1 generic argument
  //~| ERROR associated type takes 1 lifetime argument but 0 lifetime arguments
  //~| ERROR associated type takes 0 generic arguments but 1 generic argument
  //~| ERROR associated type takes 1 lifetime argument but 0 lifetime arguments
  //~| ERROR at least one trait is required
  //~| ERROR: the trait `X` cannot be made into an object


fn bar<'a>(arg: Box<dyn X<Y() = ()>>) {}
  //~^ ERROR: parenthesized generic arguments cannot be used
  //~| ERROR associated type takes 1 lifetime argument but 0 lifetime arguments
  //~| ERROR associated type takes 1 lifetime argument but 0 lifetime arguments
  //~| ERROR associated type takes 1 lifetime argument but 0 lifetime arguments
  //~| ERROR: the trait `X` cannot be made into an object

fn main() {}
