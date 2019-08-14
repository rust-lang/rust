// run-pass
#![feature(box_syntax)]

fn test_generic<T: Clone, F>(expected: Box<T>, eq: F) where F: FnOnce(Box<T>, Box<T>) -> bool {
    let actual: Box<T> = match true {
        true => { expected.clone() },
        _ => panic!("wat")
    };
    assert!(eq(expected, actual));
}

fn test_box() {
    fn compare_box(b1: Box<bool>, b2: Box<bool>) -> bool {
        return *b1 == *b2;
    }
    test_generic::<bool, _>(box true, compare_box);
}

pub fn main() { test_box(); }
