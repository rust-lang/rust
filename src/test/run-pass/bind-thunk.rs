// -*- rust -*-

fn f() -> int {
  ret 42;
}

fn main() {
  let fn() -> int g = bind f();
  let int i = g();
  check(i == 42);
}
