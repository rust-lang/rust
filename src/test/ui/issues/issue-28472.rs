// Check that the visibility modifier is included in the span of foreign items.

extern "C" {
  fn foo();

  pub //~ ERROR the name `foo` is defined multiple times
  fn foo();

  pub //~ ERROR the name `foo` is defined multiple times
  static mut foo: u32;
}

fn main() {
}
