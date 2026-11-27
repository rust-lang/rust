//@ run-rustfix
#![allow(dead_code)]

fn swap(arr: &mut [u32; 2]) {
    std::mem::swap(&mut arr[0], &mut arr[1]);
    //~^ ERROR cannot borrow `arr[_]` as mutable more than once at a time
}

fn main() {}
