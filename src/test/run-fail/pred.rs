// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

// error-pattern:predicate check

fn f(int a, int b) : lt(a,b) {
}

fn lt(int a, int b) -> bool {
  ret a < b;
}

fn main() {
  let int a = 10;
  let int b = 23;
  check lt(b,a);
  f(b,a);
}
