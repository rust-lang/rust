// check-pass

#![feature(generic_associated_types)]
  //~^ WARNING: the feature `generic_associated_types` is incomplete

trait X {
    type Y<'a>;
}

fn _func1<'a>(_x: Box<dyn X<Y<'a>=&'a ()>>) {}

fn main() {}
