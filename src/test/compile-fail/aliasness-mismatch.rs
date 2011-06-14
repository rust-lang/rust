// -*- rust -*-
// error-pattern: mismatched types

fn f(&int x) { log_err x; }
fn h(int x) { log_err x; }
fn main() {
  let fn(int x) g = f;
  g(10);
  g = h;
  g(10);
}


