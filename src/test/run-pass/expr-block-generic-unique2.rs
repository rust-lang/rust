// xfail-fast
// -*- rust -*-
#[legacy_modes];

type compare<T> = fn@(T, T) -> bool;

fn test_generic<T: Copy>(expected: T, eq: compare<T>) {
    let actual: T = { expected };
    assert (eq(expected, actual));
}

fn test_vec() {
    fn compare_vec(&&v1: ~int, &&v2: ~int) -> bool { return v1 == v2; }
    test_generic::<~int>(~1, compare_vec);
}

fn main() { test_vec(); }
