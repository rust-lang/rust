// xfail-stage0
// -*- rust -*-

fn main() {
  spawn child(10);
}

fn child(int i) {
   log i;
}

