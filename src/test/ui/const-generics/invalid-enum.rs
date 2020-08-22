#![feature(const_generics)]
#![allow(incomplete_features)]

#[derive(PartialEq, Eq)]
enum CompileFlag {
  A,
  B,
}

pub fn test<const CF: CompileFlag>() {}

pub fn main() {
  test::<CompileFlag::A>();
  //~^ ERROR: expected type, found variant
  //~| ERROR: wrong number of const arguments
  //~| ERROR: wrong number of type arguments
}
