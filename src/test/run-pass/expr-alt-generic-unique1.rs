

// -*- rust -*-
type compare<T> = fn@(~T, ~T) -> bool;

fn test_generic<T: copy>(expected: ~T, eq: compare<T>) {
    let actual: ~T = match true { true => { expected }, _ => fail ~"wat" };
    assert (eq(expected, actual));
}

fn test_box() {
    fn compare_box(b1: ~bool, b2: ~bool) -> bool { return *b1 == *b2; }
    test_generic::<bool>(~true, compare_box);
}

fn main() { test_box(); }
