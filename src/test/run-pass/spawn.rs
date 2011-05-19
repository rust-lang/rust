// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

fn main() {
  auto child_task = spawn child(10);
}

fn child(int i) {
   log i;
}

