// error-pattern:non-scalar cast

use std;
import std::os;

fn main() {
  log(debug, { x: 1 } as int);
}
