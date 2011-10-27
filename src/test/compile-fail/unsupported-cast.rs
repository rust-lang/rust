// error-pattern:unsupported cast

use std;
import std::os;

fn main() {
  log 1.0 as os::libc::FILE; // Can't cast float to native.
}
