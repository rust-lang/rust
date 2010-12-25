use std;
use libc();
use zed(name = "std");
use bar(name = "std", ver = "0.0.1");

import std._str;
import x = std._str;


mod baz {
  use std;
  use libc();
  use zed(name = "std");
  use bar(name = "std", ver = "0.0.1");

  import std._str;
  import x = std._str;
}

fn main() {
}
