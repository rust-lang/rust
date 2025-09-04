#![allow(non_camel_case_types)]
// Verifies that labels are propagated through loads and stores.
//
//@ needs-sanitizer-support
//@ needs-sanitizer-dataflow
//@ run-pass
//@ compile-flags: -Zsanitizer=dataflow -Zsanitizer-dataflow-abilist={{src-base}}/sanitizer/dataflow-abilist.txt -C unsafe-allow-abi-mismatch=sanitizer

use std::mem::size_of;
use std::os::raw::{c_int, c_long, c_void};

type dfsan_label = u8;

extern "C" {
    fn dfsan_add_label(label: dfsan_label, addr: *mut c_void, size: usize);
    fn dfsan_get_label(data: c_long) -> dfsan_label;
    fn dfsan_has_label(label: dfsan_label, elem: dfsan_label) -> c_int;
    fn dfsan_read_label(addr: *const c_void, size: usize) -> dfsan_label;
    fn dfsan_set_label(label: dfsan_label, addr: *mut c_void, size: usize);
}

fn propagate2(i: &i64) -> i64 {
    i.clone()
}

fn propagate(i: i64) -> i64 {
    let v = vec!(i, 1, 2, 3);
    let j = v.iter().sum();
    propagate2(&j)
}

pub fn main() {
    let mut i = 1i64;
    let i_ptr = &mut i as *mut i64;
    let i_label: dfsan_label = 1;
    unsafe {
        dfsan_set_label(i_label, i_ptr as *mut c_void, size_of::<i64>());
    }

    let new_label = unsafe { dfsan_get_label(i) };
    assert_eq!(i_label, new_label);

    let read_label = unsafe { dfsan_read_label(i_ptr as *const c_void, size_of::<i64>()) };
    assert_eq!(i_label, read_label);

    let j_label: dfsan_label = 2;
    unsafe {
        dfsan_add_label(j_label, i_ptr as *mut c_void, size_of::<i64>());
    }

    let read_label = unsafe { dfsan_read_label(i_ptr as *const c_void, size_of::<i64>()) };
    assert_eq!(unsafe { dfsan_has_label(read_label, i_label) }, 1);
    assert_eq!(unsafe { dfsan_has_label(read_label, j_label) }, 1);

    let mut new_i = propagate(i);
    let new_i_ptr = &mut new_i as *mut i64;
    let read_label = unsafe { dfsan_read_label(new_i_ptr as *const c_void, size_of::<i64>()) };
    assert_eq!(unsafe { dfsan_has_label(read_label, i_label) }, 1);
    assert_eq!(unsafe { dfsan_has_label(read_label, j_label) }, 1);
}
