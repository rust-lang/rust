// error-pattern:unsupported cast

use std;
import std::os;

fn main() {
  log(debug, 1.0 as os::FILE); // Can't cast float to native.
}
