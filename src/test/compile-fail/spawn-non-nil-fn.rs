// xfail-stage0
// xfail-stage1
// xfail-stage2
// error-pattern: mismatched types

fn f(int x) -> int {
  ret x;
}

fn main() {
  spawn f(10);
}
