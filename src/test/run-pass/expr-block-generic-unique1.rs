#![feature(box_syntax)]

fn test_generic<T, F>(expected: Box<T>, eq: F) where T: Clone, F: FnOnce(Box<T>, Box<T>) -> bool {
    let actual: Box<T> = { expected.clone() };
    assert!(eq(expected, actual));
}

fn test_box() {
    fn compare_box(b1: Box<bool>, b2: Box<bool>) -> bool {
        println!("{}", *b1);
        println!("{}", *b2);
        return *b1 == *b2;
    }
    test_generic::<bool, _>(box true, compare_box);
}

pub fn main() { test_box(); }
