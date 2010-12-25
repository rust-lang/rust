use std;
use libc();
use zed(name = "std");
use bar(name = "std", ver = "0.0.1");

mod baz {
  use std;
  use libc();
  use zed(name = "std");
  use bar(name = "std", ver = "0.0.1");
}

fn main() {
}
