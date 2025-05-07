trait X {
  type Y<'x>;
}

fn main() {
  fn _f(arg : Box<dyn for<'a> X<Y<'x> = &'a [u32]>>) {}
    //~^ ERROR: use of undeclared lifetime name `'x`
    //~| ERROR: binding for associated type `Y` references lifetime
    //~| ERROR: binding for associated type `Y` references lifetime
    //~| ERROR: binding for associated type `Y` references lifetime
    //~| ERROR: the trait `X` is not dyn compatible
}
