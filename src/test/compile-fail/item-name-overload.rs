// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

// error-pattern: name

mod foo {
  fn bar[T](T f) -> int { ret 17; }
  type bar[U, T] = tup(int, U, T);
}

fn main() {}
