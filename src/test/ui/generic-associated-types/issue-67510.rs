#![feature(generic_associated_types)]
  //~^ WARNING: the feature `generic_associated_types` is incomplete

trait X {
    type Y<'a>;
}

fn f(x: Box<dyn X<Y<'a>=&'a ()>>) {}
  //~^ ERROR: use of undeclared lifetime name `'a`
  //~| ERROR: use of undeclared lifetime name `'a`


fn main() {}
