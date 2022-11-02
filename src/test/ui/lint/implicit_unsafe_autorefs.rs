// check-pass
// run-rustfix
#![allow(dead_code)]
use std::ptr::{addr_of, addr_of_mut};

unsafe fn test_mut(ptr: *mut [u8]) -> *mut [u8] {
    addr_of_mut!((*ptr)[..16])
    //~^ warn: implicit auto-ref creates a reference to a dereference of a raw pointer
}

unsafe fn test_const(ptr: *const [u8]) -> *const [u8] {
    addr_of!((*ptr)[..16])
    //~^ warn: implicit auto-ref creates a reference to a dereference of a raw pointer
}

struct Test {
    field: [u8],
}

unsafe fn test_field(ptr: *const Test) -> *const [u8] {
    let l = (*ptr).field.len();
    //~^ warn: implicit auto-ref creates a reference to a dereference of a raw pointer

    addr_of!((*ptr).field[..l - 1])
    //~^ warn: implicit auto-ref creates a reference to a dereference of a raw pointer
}

fn main() {}
