#![crate_type="lib"]
pub struct Foo(());

impl Foo {
  pub fn new() -> Foo {
    Foo(())
  }
}
