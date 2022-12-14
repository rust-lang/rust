trait X {
  type Y<'a>;

  fn foo<'a>(t : Self::Y<'a>) -> Self::Y<'a> { t }
}

impl<T> X for T {
  fn foo<'a, T1: X<Y = T1>>(t : T1) -> T1::Y<'a> {
    //~^ ERROR missing generics for associated type
    //~^^ ERROR missing generics for associated type
    t
  }
}

fn main() {}
