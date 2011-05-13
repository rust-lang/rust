// xfail-stage0
// error-pattern:explicit failure

fn main() {
  auto x = alt(true) {
    case (false) {
      0
    }
    case (true) {
      fail
    }
  };
}
