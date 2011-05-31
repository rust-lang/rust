// xfail-stage0

fn test_int() {
  fn f() -> int { 10 }
  assert (f() == 10);
}

fn test_vec() {
  fn f() -> vec[int] { [10, 11] }
  assert (f().(1) == 11);
}

fn test_generic() {
  fn f[T](&T t) -> T { t }
  assert (f(10) == 10);
}

fn test_alt() {
  fn f() -> int {
    alt (true) {
      case (false) { 10 }
      case (true) { 20 }
    }
  }
  assert (f() == 20);
}

fn test_if() {
  fn f() -> int { if (true) { 10 } else { 20 } }
  assert (f() == 10);
}

fn test_block() {
  fn f() -> int {{ 10 }}
  assert (f() == 10);
}

fn test_ret() {
  fn f() -> int {
    ret 10 // no semi
  }
  assert (f() == 10);
}

// From issue #372
fn test_372() {
  fn f() -> int { auto x = { 3 }; x }
  assert (f() == 3);
}

fn test_nil() {
  ()
}

fn main() {
  test_int();
  test_vec();
  test_generic();
  test_alt();
  test_if();
  test_block();
  test_ret();
  test_372();
  test_nil();
}
