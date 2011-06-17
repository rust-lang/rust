// xfail-stage0
// error-pattern:mismatched types
// issue #500

fn main() {
  auto x = true;
  auto y = 1;
  auto z = x + y;
}
