// error-pattern:f is not a variant
import bad::*;

mod bad {
  export b::{f, z};

  enum b { z, k }
  fn f() { fail; }
}

fn main() {
}