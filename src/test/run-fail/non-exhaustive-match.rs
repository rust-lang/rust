// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

// error-pattern:non-exhaustive match failure

tag t {
  a;
  b;
}

fn main() {
  auto x = a;
  alt (x) {
    case (b) { }
  }
}
