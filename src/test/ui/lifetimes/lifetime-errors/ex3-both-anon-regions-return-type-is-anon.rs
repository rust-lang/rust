struct Foo {
  field: i32
}

impl Foo {
  fn foo<'a>(&self, x: &i32) -> &i32 {
    x //~ ERROR lifetime mismatch
  }
}

fn main() { }
