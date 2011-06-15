


// -*- rust -*-
type compare[T] = fn(@T, @T) -> bool ;

fn test_generic[T](@T expected, @T not_expected, &compare[T] eq) {
    let @T actual = if (true) { expected } else { not_expected };
    assert (eq(expected, actual));
}

fn test_box() {
    fn compare_box(@bool b1, @bool b2) -> bool { ret *b1 == *b2; }
    auto eq = bind compare_box(_, _);
    test_generic[bool](@true, @false, eq);
}

fn main() { test_box(); }