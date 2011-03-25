// xfail-boot
// error-pattern: Non-call expression in tail call

fn f() -> int {
  auto x = 1;
  be x;
}

fn main() {
  auto y = f();
}
