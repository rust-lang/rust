


// -*- rust -*-
type compare[T] = fn(@T, @T) -> bool ;

fn test_generic[T](expected: @T, not_expected: @T, eq: &compare[T]) {
    let actual: @T = if true { expected } else { not_expected };
    assert (eq(expected, actual));
}

fn test_box() {
    fn compare_box(b1: @bool, b2: @bool) -> bool { ret *b1 == *b2; }
    let eq = bind compare_box(_, _);
    test_generic[bool](@true, @false, eq);
}

fn main() { test_box(); }