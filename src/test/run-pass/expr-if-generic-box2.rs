


// -*- rust -*-
type compare[T] = fn(&T, &T) -> bool ;

fn test_generic[T](&T expected, &T not_expected, &compare[T] eq) {
    let T actual = if (true) { expected } else { not_expected };
    assert (eq(expected, actual));
}

fn test_vec() {
    fn compare_vec(&vec[int] v1, &vec[int] v2) -> bool { ret v1 == v2; }
    auto eq = bind compare_vec(_, _);
    test_generic[vec[int]]([1, 2], [2, 3], eq);
}

fn main() { test_vec(); }