fn test_generic<T, F>(expected: T, not_expected: T, eq: F) where
    T: Clone,
    F: FnOnce(T, T) -> bool,
{
    let actual: T = if true { expected.clone() } else { not_expected };
    assert!(eq(expected, actual));
}

fn test_bool() {
    fn compare_bool(b1: bool, b2: bool) -> bool { return b1 == b2; }
    test_generic::<bool, _>(true, false, compare_bool);
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
    test_generic::<Pair, _>(Pair{a: 1, b: 2}, Pair{a: 2, b: 3}, compare_rec);
}

pub fn main() { test_bool(); test_rec(); }
