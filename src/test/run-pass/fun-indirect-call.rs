// -*- rust -*-

fn f() -> int {
  ret 42;
}

fn main() {
  let fn() -> int g = f;
  let int i = g();
  check(i == 42);
}
