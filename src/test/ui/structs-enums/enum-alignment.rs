// run-pass
#![allow(dead_code)]
#![allow(deprecated)]

use std::mem;

fn addr_of<T>(ptr: &T) -> usize {
    ptr as *const T as usize
}

fn is_aligned<T>(ptr: &T) -> bool {
    unsafe {
        let addr: usize = mem::transmute(ptr);
        (addr % mem::min_align_of::<T>()) == 0
    }
}

pub fn main() {
    let x = Some(0u64);
    match x {
        None => panic!(),
        Some(ref y) => assert!(is_aligned(y))
    }
}
