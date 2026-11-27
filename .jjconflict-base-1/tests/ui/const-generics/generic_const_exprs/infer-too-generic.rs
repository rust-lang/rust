//@ run-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use std::{mem, ptr};

fn split_first<T, const N: usize>(arr: [T; N]) -> (T, [T; N - 1])
where
    [T; N - 1]: Sized,
{
    let arr = mem::ManuallyDrop::new(arr);
    unsafe {
        let head = ptr::read(&arr[0]);
        let tail = ptr::read(&arr[1..] as *const [T] as *const [T; N - 1]);
        (head, tail)
    }
}

fn main() {
    let arr = [0, 1, 2, 3, 4];
    let (head, tail) = split_first(arr);
    assert_eq!(head, 0);
    assert_eq!(tail, [1, 2, 3, 4]);
}
