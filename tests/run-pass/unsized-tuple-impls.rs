#![feature(unsized_tuple_coercion)]
use std::mem;

fn main() {
    let x : &(i32, i32, [i32]) = &(0, 1, [2, 3]);
    let y : &(i32, i32, [i32]) = &(0, 1, [2, 3, 4]);
    let mut a = [y, x];
    a.sort();
    assert_eq!(a, [x, y]);

    assert_eq!(&format!("{:?}", a), "[(0, 1, [2, 3]), (0, 1, [2, 3, 4])]");
    assert_eq!(mem::size_of_val(x), 16);
}
