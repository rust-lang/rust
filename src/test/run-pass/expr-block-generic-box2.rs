


// -*- rust -*-
type compare[T] = fn(&T, &T) -> bool ;

fn test_generic[T](&T expected, &compare[T] eq) {
    let T actual = { expected };
    assert (eq(expected, actual));
}

fn test_vec() {
    fn compare_vec(&vec[int] v1, &vec[int] v2) -> bool { ret v1 == v2; }
    auto eq = bind compare_vec(_, _);
    test_generic[vec[int]]([1, 2], eq);
}

fn main() { test_vec(); }