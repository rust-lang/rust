struct Foo {
  field: i32
}

impl Foo {
  fn foo<'a>(&'a self, x: &i32) -> &i32 {

    if true { &self.field } else { x } //~ ERROR explicit lifetime

  }

}

fn main() { }
