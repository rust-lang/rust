#![feature(type_ascription)]

use std::convert::AsRef;

struct Foo {
  a : u32,
}

impl AsRef<Foo> for Foo {
  fn as_ref(&self) -> &Foo {
    &self
  }
}

fn main() {
  let foo = Foo { a : 1 };
  let r = &mut foo;

  let x = (r : &Foo).as_ref();
    //~^ ERROR: type ascriptions are not 

  let another_one = (r : &Foo).as_ref().a;
    //~^ ERROR: type ascriptions are not 

  let last_one = &*((r : &Foo).as_ref());
    //~^ ERROR: type ascriptions are not 
}
