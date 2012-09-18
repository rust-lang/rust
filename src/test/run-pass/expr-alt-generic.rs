// -*- rust -*-
#[legacy_modes];

type compare<T> = fn@(T, T) -> bool;

fn test_generic<T: Copy>(expected: T, eq: compare<T>) {
  let actual: T = match true { true => { expected }, _ => fail ~"wat" };
    assert (eq(expected, actual));
}

fn test_bool() {
    fn compare_bool(&&b1: bool, &&b2: bool) -> bool { return b1 == b2; }
    test_generic::<bool>(true, compare_bool);
}

type t = {a: int, b: int};

fn test_rec() {
    fn compare_rec(t1: t, t2: t) -> bool {
        t1.a == t2.a && t1.b == t2.b
    }
    test_generic::<t>({a: 1, b: 2}, compare_rec);
}

fn main() { test_bool(); test_rec(); }
