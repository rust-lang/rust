// xfail-stage0
// error-pattern: mismatched types

fn f(int x) -> int {
  ret x;
}

fn main() {
  spawn f(10);
}
