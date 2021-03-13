#![feature(generic_associated_types)]
  //~^ WARNING: the feature `generic_associated_types` is incomplete

trait X {
  type Y<'a>;
    //~^ ERROR missing generics for
    //~| ERROR missing generics for

  fn foo<'a>(t : Self::Y<'a>) -> Self::Y<'a> { t }
}

impl<T> X for T {
  fn foo<'a, T1: X<Y = T1>>(t : T1) -> T1::Y<'a> {
    t
  }
}

fn main() {}
