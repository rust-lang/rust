// xfail-stage0

// When all branches of an if expression result in fail, the entire if
// expression results in fail.

fn main() {
  auto x = if (true) {
    10
  } else {
    if (true) {
      fail
    } else {
      fail
    }
  };
}
