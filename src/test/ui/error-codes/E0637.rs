struct Foo<'a: '_>(&'a u8); //~ ERROR invalid lifetime bound name: `'_`
fn foo<'a: '_>(_: &'a u8) {} //~ ERROR invalid lifetime bound name: `'_`

struct Bar<'a>(&'a u8);
impl<'a: '_> Bar<'a> { //~ ERROR invalid lifetime bound name: `'_`
  fn bar() {}
}

fn main() {}
