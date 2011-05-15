// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
// -*- rust -*-

// error-pattern: may not return

fn god_exists(int a) -> bool {
  be god_exists(a);
}

fn f(int a) -> int {
  if (god_exists(a)) {
    ret 5;
  }
}

fn main() {
  f(12);
}
