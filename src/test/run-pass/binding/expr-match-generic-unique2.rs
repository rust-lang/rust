// run-pass
#![feature(box_syntax)]

fn test_generic<T: Clone, F>(expected: T, eq: F) where F: FnOnce(T, T) -> bool {
    let actual: T = match true {
        true => expected.clone(),
        _ => panic!("wat")
    };
    assert!(eq(expected, actual));
}

fn test_vec() {
    fn compare_box(v1: Box<isize>, v2: Box<isize>) -> bool { return v1 == v2; }
    test_generic::<Box<isize>, _>(box 1, compare_box);
}

pub fn main() { test_vec(); }
