// xfail-stage0
// xfail-stage1
// xfail-stage2

fn main() {
  auto x = if (false) {
    [0u]
  } else if (true) {
    [10u]
  } else {
    [0u]
  };
}
