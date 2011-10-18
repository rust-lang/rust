


// -*- rust -*-

// Tests for if as expressions with dynamic type sizes
type compare<T> = fn@(T, T) -> bool;

fn test_generic<@T>(expected: T, not_expected: T, eq: compare<T>) {
    let actual: T = if true { expected } else { not_expected };
    assert (eq(expected, actual));
}

fn test_bool() {
    fn compare_bool(&&b1: bool, &&b2: bool) -> bool { ret b1 == b2; }
    let eq = bind compare_bool(_, _);
    test_generic::<bool>(true, false, eq);
}

fn test_rec() {
    type t = {a: int, b: int};

    fn compare_rec(t1: t, t2: t) -> bool { ret t1 == t2; }
    let eq = bind compare_rec(_, _);
    test_generic::<t>({a: 1, b: 2}, {a: 2, b: 3}, eq);
}

fn main() { test_bool(); test_rec(); }
