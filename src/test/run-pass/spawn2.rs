// xfail-stage0
// -*- rust -*-

fn main() {
  spawn child(10, 20);
}

fn child(int i, int j) {
  log_err i;
  log_err j;
}
