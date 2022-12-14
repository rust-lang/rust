struct Foo;
impl Foo {
  fn bar() { foobar::ok(); }
}

fn main() {
  Foo::bar();
}
