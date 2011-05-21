// -*- rust -*-
// xfail-stage0
// error-pattern: not all control paths return

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
