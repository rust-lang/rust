


// -*- rust -*-
type compare<T> = fn@(T, T) -> bool;

fn test_generic<@T>(expected: T, eq: compare<T>) {
    let actual: T = { expected };
    assert (eq(expected, actual));
}

fn test_vec() {
    fn compare_vec(&&v1: ~int, &&v2: ~int) -> bool { ret v1 == v2; }
    let eq = bind compare_vec(_, _);
    test_generic::<~int>(~1, eq);
}

fn main() { test_vec(); }
