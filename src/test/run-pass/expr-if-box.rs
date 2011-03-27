// xfail-boot
// xfail-stage0
// -*- rust -*-

// Tests for if as expressions returning boxed types

fn test_box() {
  auto res = if (true) { @100 } else { @101 };
  check (*res == 100);
}

fn main() {
  test_box();
}