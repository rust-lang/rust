


// -*- rust -*-
type compare<T> = fn@(@T, @T) -> bool;

fn test_generic<T>(expected: @T, eq: compare<T>) {
    let actual: @T = { expected };
    assert (eq(expected, actual));
}

fn test_box() {
    fn compare_box(b1: @bool, b2: @bool) -> bool {
        log_full(core::debug, *b1);
        log_full(core::debug, *b2);
        ret *b1 == *b2;
    }
    let eq = bind compare_box(_, _);
    test_generic::<bool>(@true, eq);
}

fn main() { test_box(); }
