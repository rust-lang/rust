


// -*- rust -*-

// Tests for if as expressions with dynamic type sizes
type compare<T> = fn@(T, T) -> bool;

fn test_generic<T: copy>(expected: T, not_expected: T, eq: compare<T>) {
    let actual: T = if true { expected } else { not_expected };
    assert (eq(expected, actual));
}

fn test_bool() {
    fn compare_bool(&&b1: bool, &&b2: bool) -> bool { return b1 == b2; }
    test_generic::<bool>(true, false, compare_bool);
}

type t = {a: int, b: int};
impl t : cmp::Eq {
    pure fn eq(&&other: t) -> bool {
        self.a == other.a && self.b == other.b
    }
}

fn test_rec() {
    fn compare_rec(t1: t, t2: t) -> bool { return t1 == t2; }
    test_generic::<t>({a: 1, b: 2}, {a: 2, b: 3}, compare_rec);
}

fn main() { test_bool(); test_rec(); }
