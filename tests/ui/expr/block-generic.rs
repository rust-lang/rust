//@ run-pass
#![allow(unused_braces)]

fn test_generic<T: Clone, F>(expected: T, eq: F) where F: FnOnce(T, T) -> bool {
    let actual: T = { expected.clone() };
    assert!(eq(expected, actual));
}

fn test_bool() {
    fn compare_bool(b1: bool, b2: bool) -> bool { return b1 == b2; }
    test_generic::<bool, _>(true, compare_bool);
}

#[derive(Clone)]
struct Pair {
    a: isize,
    b: isize,
}

fn test_rec() {
    fn compare_rec(t1: Pair, t2: Pair) -> bool {
        t1.a == t2.a && t1.b == t2.b
    }
    test_generic::<Pair, _>(Pair {a: 1, b: 2}, compare_rec);
}

pub fn main() { test_bool(); test_rec(); }
