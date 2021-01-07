#![feature(type_ascription)]

struct Foo {}

impl Foo {
  fn foo(&self) {}
}

fn main() {
  let mut fooer = Foo {};
  let x = &mut fooer;
  (x : &Foo).foo(); 
    //~^ ERROR: type ascriptions are not allowed
}
