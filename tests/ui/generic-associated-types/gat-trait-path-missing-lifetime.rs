trait X {
  type Y<'a>;

  fn foo<'a>(t : Self::Y<'a>) -> Self::Y<'a> { t }
}

impl<T> X for T { //~ ERROR: not all trait items implemented
  fn foo<'a, T1: X<Y = T1>>(t : T1) -> T1::Y<'a> {
    //~^ ERROR missing generics for associated type
    //~^^ ERROR missing generics for associated type
    //~| ERROR associated function `foo` has 1 type parameter but its trait declaration has 0 type parameters
    t
  }
}

fn main() {}
