// xfail-stage0
// -*- rust -*-

use std;

fn main() {
  let int i = 0;
  while (i < 100) {
    i = i + 1;
    log_err i;
    std::task::yield();
  }
}
