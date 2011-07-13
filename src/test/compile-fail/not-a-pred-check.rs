// -*- rust -*-
// error-pattern: non-predicate

fn f(int q) -> bool { ret true; }

fn main() {
  auto x = 0;

  check f(x);
}
