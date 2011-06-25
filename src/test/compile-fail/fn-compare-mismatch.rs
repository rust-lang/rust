// xfail-stage0
// error-pattern:expected fn() but found fn(int)

fn main() {
  fn f() {}
  fn g(int i) {}
  auto x = f == g;
}