// -*- rust -*-

// error-pattern: expected the constraint name

obj f () {
  fn g (int q) -> bool {
    ret true;
  }
}

fn main() {
  auto z = f ();
  check (z.g)(42); // should fail to typecheck, as z.g isn't an explicit name
}
