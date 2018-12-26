struct Foo {
  field: i32
}

impl Foo {
  fn foo<'a>(&self, x: &'a i32) -> &i32 {

    x //~ ERROR lifetime mismatch

  }

}

fn main() { }
