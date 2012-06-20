


// -*- rust -*-
type compare<T> = fn@(T, T) -> bool;

fn test_generic<T: copy>(expected: T, eq: compare<T>) {
    let actual: T = alt check true { true { expected } };
    assert (eq(expected, actual));
}

fn test_vec() {
    fn compare_box(&&v1: @int, &&v2: @int) -> bool { ret v1 == v2; }
    test_generic::<@int>(@1, compare_box);
}

fn main() { test_vec(); }
