// -*- rust -*-

// error-pattern: attempted dynamic environment-capture

fn f(bool x) {
}

state obj foobar(bool x) {
  drop {
    auto y = x;
    fn test() {
      f(y);
    }
  }
}

fn main() {
}
