// xfail-boot
// -*- rust -*-

// Tests for using alt as an expression

fn test() {
  let bool res = alt (true) {
    case (true) {
      true
    }
    case (false) {
      false
    }
  };
  check (res);

  res = alt(false) {
    case (true) {
      false
    }
    case (false) {
      true
    }
  };
  check (res);
}

fn main() {
  test();
}
