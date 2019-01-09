// run-pass

use std::vec;

pub fn main() {
    let a: Vec<isize> = vec![1, 2, 3, 4, 5];
    let b: Vec<isize> = vec![6, 7, 8, 9, 0];
    let mut v: Vec<isize> = a;
    v.extend_from_slice(&b);
    println!("{}", v[9]);
    assert_eq!(v[0], 1);
    assert_eq!(v[7], 8);
    assert_eq!(v[9], 0);
}
