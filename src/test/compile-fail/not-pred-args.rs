// -*- rust -*-

// error-pattern: Constraint args must be

fn f(int q) -> bool { ret true; }

fn main() {
// should fail to typecheck, as pred args must be slot variables or literals
  check f(42 * 17); 
}
