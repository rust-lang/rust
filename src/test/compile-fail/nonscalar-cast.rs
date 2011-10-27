// error-pattern:non-scalar cast

use std;
import std::os;

fn main() {
  log { x: 1 } as int;
}
