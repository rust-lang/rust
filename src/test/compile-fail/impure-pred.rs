// -*- rust -*-
// error-pattern: impure function

fn g() -> () {}

pred f(int q) -> bool { 
  g();
  ret true;
}

fn main() {
  auto x = 0;

  check f(x); 
}
