// run-pass

#![allow(dead_code)]
#![allow(unused_variables)]
// Type ascription doesn't lead to unsoundness

#![feature(type_ascription)]

use std::mem;

const C1: u8 = 10: u8;
const C2: [u8; 1: usize] = [1];

struct S {
    a: u8
}

fn main() {
    assert_eq!(C1.into(): i32, 10);
    assert_eq!(C2[0], 1);

    let s = S { a: 10: u8 };
    let arr = &[1u8, 2, 3];

    let mut v = arr.iter().cloned().collect(): Vec<_>;
    v.push(4);
    assert_eq!(v, [1, 2, 3, 4]);

    let a = 1: u8;
    let b = a.into(): u16;
    assert_eq!(v[a.into(): usize], 2);
    assert_eq!(mem::size_of_val(&a), 1);
    assert_eq!(mem::size_of_val(&b), 2);
    assert_eq!(b, 1: u16);

    let mut v = Vec::new();
    v: Vec<u8> = vec![1, 2, 3]; // Place expression type ascription
    assert_eq!(v, [1u8, 2, 3]);
}
