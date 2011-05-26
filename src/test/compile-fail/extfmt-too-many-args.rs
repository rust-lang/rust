// xfail-stage0
// error-pattern:too many arguments

use std;

fn main() {
  auto s = #fmt("%s", "test", "test");
}
