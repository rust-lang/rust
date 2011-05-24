// xfail-stage0
// -*- rust -*-

fn main() {
  auto t = spawn child(10);
}

fn child(int i) {
   log_err i;
}

