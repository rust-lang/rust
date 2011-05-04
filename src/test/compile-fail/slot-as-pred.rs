// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

// error-pattern: mismatched types

fn f(int a, int b) : lt(a,b) {
}

fn main() {
  let int lt;
  let int a = 10;
  let int b = 23;
  check lt(a,b);
  f(a,b);
}
