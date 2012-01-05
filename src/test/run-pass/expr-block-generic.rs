


// -*- rust -*-

// Tests for standalone blocks as expressions with dynamic type sizes
type compare<T> = fn@(T, T) -> bool;

fn test_generic<T: copy>(expected: T, eq: compare<T>) {
    let actual: T = { expected };
    assert (eq(expected, actual));
}

fn test_bool() {
    fn compare_bool(&&b1: bool, &&b2: bool) -> bool { ret b1 == b2; }
    let eq = bind compare_bool(_, _);
    test_generic::<bool>(true, eq);
}

fn test_rec() {
    type t = {a: int, b: int};

    fn compare_rec(t1: t, t2: t) -> bool { ret t1 == t2; }
    let eq = bind compare_rec(_, _);
    test_generic::<t>({a: 1, b: 2}, eq);
}

fn main() { test_bool(); test_rec(); }
