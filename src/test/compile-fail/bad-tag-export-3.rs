// error-pattern:b does not refer to an enumeration
import bad::*;

mod bad {
  export b::{f, z};

  fn b() { fail; }
  fn f() { fail; }
  fn z() { fail; }
}

fn main() {
}