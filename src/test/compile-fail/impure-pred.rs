// -*- rust -*-
// error-pattern: Pure function calls function not known to be pure

fn g() -> () {}

pred f(int q) -> bool {
  g();
  ret true;
}

fn main() {
  auto x = 0;

  check f(x);
}
