// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

fn main() {
  auto t = spawn child(10);
}

fn child(int i) {
   log_err i;
}

