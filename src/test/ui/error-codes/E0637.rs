struct Foo<'a: '_>(&'a u8); //~ ERROR cannot be used here
fn foo<'a: '_>(_: &'a u8) {} //~ ERROR cannot be used here

struct Bar<'a>(&'a u8);
impl<'a: '_> Bar<'a> { //~ ERROR cannot be used here
  fn bar() {}
}

fn main() {}
