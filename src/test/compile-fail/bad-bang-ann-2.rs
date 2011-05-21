// -*- rust -*-
// xfail-stage0
// Tests that a function with a ! annotation always actually fails
// error-pattern: some control paths may return

fn bad_bang(uint i) -> ! {
  log 3;
}

fn main() {
  bad_bang(5u);
}