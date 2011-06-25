// xfail-stage0
// -*- rust -*-

// error-pattern: Unsatisfied precondition constraint (for example, lt(a, b)

fn f(int a, int b) : lt(a,b) {
}

pred lt(int a, int b) -> bool {
  ret a < b;
}

fn main() {
  let int a = 10;
  let int b = 23;
  let int c = 77;
  check lt(a,b);
  b <-> a;
  f(a,b);
}
