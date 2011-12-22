// error-pattern:non-scalar cast

use std;
import std::os;

fn main() {
  log_full(core::debug, { x: 1 } as int);
}
