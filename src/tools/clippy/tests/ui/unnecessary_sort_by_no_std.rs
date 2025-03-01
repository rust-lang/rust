#![no_std]
extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

fn issue_11524() -> Vec<i32> {
    let mut vec = vec![1, 2, 3];

    // Should lint and suggest `vec.sort_by_key(|a| a + 1);`
    vec.sort_by(|a, b| (a + 1).cmp(&(b + 1)));
    //~^ unnecessary_sort_by
    vec
}

fn issue_11524_2() -> Vec<i32> {
    let mut vec = vec![1, 2, 3];

    // Should lint and suggest `vec.sort_by_key(|b| core::cmp::Reverse(b + 1));`
    vec.sort_by(|a, b| (b + 1).cmp(&(a + 1)));
    //~^ unnecessary_sort_by
    vec
}
