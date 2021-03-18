#![feature(generic_associated_types)]
  //~^ the feature `generic_associated_types` is incomplete

trait X {
    type Y<'a>;
      //~^ ERROR this associated type
      //~| ERROR this associated type
}

const _: () = {
  fn f2<'a>(arg : Box<dyn X<Y<1> = &'a ()>>) {}
};

fn main() {}
