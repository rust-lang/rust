// -*- rust -*-
// xfail-stage0
// error-pattern: Dynamically sized arguments must be passed by alias

mod foo {
  fn bar[T](T f) -> int { ret 17; }
  type bar[U, T] = rec(int a, U b, T c);
}

fn main() {}
