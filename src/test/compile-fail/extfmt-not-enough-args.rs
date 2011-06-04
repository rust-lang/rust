// xfail-stage0
// error-pattern:not enough arguments

use std;

fn main() {
  auto s = #fmt("%s%s%s", "test", "test");
}
