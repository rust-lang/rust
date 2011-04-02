// xfail-boot
// -*- rust -*-

// Tests for alt as expressions resulting in structural types

fn test_rec() {
  auto res = alt (true) {
    case (true) {
      rec(i = 100)
    }
  };
  check (res == rec(i = 100));
}

fn test_tag() {
  tag mood {
    happy;
    sad;
  }

  auto res = alt (true) {
    case (true) {
      happy
    }
    case (false) {
      sad
    }
  };
  check (res == happy);
}

fn main() {
  test_rec();
  test_tag();
}