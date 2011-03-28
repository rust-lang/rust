// xfail-boot
// xfail-stage0
// -*- rust -*-

// Tests for if as expressions with dynamic type sizes

type compare[T] = fn(&T t1, &T t2) -> bool;

fn test_generic[T](&T expected, &T not_expected, &compare[T] eq) {
  let T actual = if (true) { expected } else { not_expected };
  check (eq(expected, actual));
}

fn test_bool() {
  fn compare_bool(&bool b1, &bool b2) -> bool {
    ret b1 == b2;
  }
  auto eq = bind compare_bool(_, _);
  test_generic[bool](true, false, eq);
}

fn test_tup() {
  type t = tup(int, int);
  fn compare_tup(&t t1, &t t2) -> bool {
    ret t1 == t2;
  }
  auto eq = bind compare_tup(_, _);
  test_generic[t](tup(1, 2), tup(2, 3), eq);
}

fn test_vec() {
  fn compare_vec(&vec[int] v1, &vec[int] v2) -> bool {
    ret v1 == v2;
  }
  auto eq = bind compare_vec(_, _);
  test_generic[vec[int]](vec(1, 2), vec(2, 3), eq);
}

fn test_box() {
  fn compare_box(&@bool b1, &@bool b2) -> bool {
    ret *b1 == *b2;
  }
  auto eq = bind compare_box(_, _);
  test_generic[@bool](@true, @false, eq);
}

fn main() {
  test_bool();
  test_tup();
  // FIXME: These two don't pass yet
  test_vec();
  test_box();
}