// error-pattern:explicit failure

fn main() {
  auto x = if (false) {
    0
  } else if (true) {
    fail
  } else {
    10
  };
}
