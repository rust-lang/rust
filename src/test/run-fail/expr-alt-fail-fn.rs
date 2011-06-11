// error-pattern:explicit failure

fn f() -> ! { fail }

fn g() -> int {
  auto x = alt (true) {
    case (true) {
      f()
    }
    case (false) {
      10
    }
  };
  ret x;
}

fn main() {
  g();
}
