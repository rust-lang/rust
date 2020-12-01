#![feature(generic_associated_types)]
//~^ WARNING: the feature `generic_associated_types` is incomplete

mod error1 {
  trait X {
      type Y<'a>;
  }

  fn f1<'a>(arg : Box<dyn X< 1 = 32 >>) {}
      //~^ ERROR: expected expression, found `)`
}

mod error2 {

  trait X {
      type Y<'a>;
  }

  fn f2<'a>(arg : Box<dyn X< { 1 } = 32 >>) {}
    //~^ ERROR: only types can be used in associated type constraints
}

fn main() {}
