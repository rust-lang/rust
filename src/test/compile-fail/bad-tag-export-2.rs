// error-pattern:b does not refer to an enumeration
import bad::*;

mod bad {
  export b::{};

  fn b() { fail; }
}

fn main() {
}