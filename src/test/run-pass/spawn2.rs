// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

use std;

fn main() {
  spawn child(10, 20);
}

fn child(int i, int j) {
  log_err i;
  log_err j;
}
