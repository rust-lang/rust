// -*- rust -*-
// xfail-stage0
// Tests that a function with a ! annotation always actually fails
// error-pattern: may return to the caller

fn bad_bang(uint i) -> ! {
  if (i < 0u) {
  }
  else {
    fail;
  }
}

fn main() {
  bad_bang(5u);
}