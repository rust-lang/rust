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
    //~^ ERROR: expected one of
}

fn main() {}
