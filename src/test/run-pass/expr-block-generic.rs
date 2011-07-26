


// -*- rust -*-

// Tests for standalone blocks as expressions with dynamic type sizes
type compare[T] = fn(&T, &T) -> bool ;

fn test_generic[T](&T expected, &compare[T] eq) {
    let T actual = { expected };
    assert (eq(expected, actual));
}

fn test_bool() {
    fn compare_bool(&bool b1, &bool b2) -> bool { ret b1 == b2; }
    auto eq = bind compare_bool(_, _);
    test_generic[bool](true, eq);
}

fn test_rec() {
    type t = rec(int a, int b);

    fn compare_rec(&t t1, &t t2) -> bool { ret t1 == t2; }
    auto eq = bind compare_rec(_, _);
    test_generic[t](rec(a=1, b=2), eq);
}

fn main() { test_bool(); test_rec(); }