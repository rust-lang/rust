// xfail-stage0
// error-pattern:expected bool but found int
// issue #516

fn main() {
  auto x = true;
  auto y = 1;
  auto z = x + y;
}