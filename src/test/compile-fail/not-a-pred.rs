// -*- rust -*-
// xfail-stage0
// error-pattern: Non-predicate in constraint: lt

fn f(int a, int b) : lt(a,b) {
}

obj lt(int a, int b) {
}

fn main() {
  let int a = 10;
  let int b = 23;
  check lt(a,b);
  f(a,b);
}
