// run-rustfix

#![allow(unused)]

use std::collections::HashSet;
use std::collections::VecDeque;

fn main() {
    let v = [1, 2, 3, 4, 5];
    let v2: Vec<isize> = v.iter().cloned().collect();
    let v3: HashSet<isize> = v.iter().cloned().collect();
    let v4: VecDeque<isize> = v.iter().cloned().collect();

    // Handle macro expansion in suggestion
    let _: Vec<isize> = vec![1, 2, 3].iter().cloned().collect();

    // Issue #3704
    unsafe {
        let _: Vec<u8> = std::ffi::CStr::from_ptr(std::ptr::null())
            .to_bytes()
            .iter()
            .cloned()
            .collect();
    }

    // Issue #6808
    let arr: [u8; 64] = [0; 64];
    let _: Vec<_> = arr.iter().cloned().collect();

    // Issue #6703
    let _: Vec<isize> = v.iter().copied().collect();
}
