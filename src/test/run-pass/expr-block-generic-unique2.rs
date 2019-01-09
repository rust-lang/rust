#![feature(box_syntax)]

fn test_generic<T, F>(expected: T, eq: F) where T: Clone, F: FnOnce(T, T) -> bool {
    let actual: T = { expected.clone() };
    assert!(eq(expected, actual));
}

fn test_vec() {
    fn compare_vec(v1: Box<isize>, v2: Box<isize>) -> bool { return v1 == v2; }
    test_generic::<Box<isize>, _>(box 1, compare_vec);
}

pub fn main() { test_vec(); }
