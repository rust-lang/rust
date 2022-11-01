// Issue #80127: Passing structs via FFI should work with explicit alignment.

use std::ffi::CString;
use std::ptr::null_mut;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[repr(align(16))]
pub struct TwoU64s {
    pub a: u64,
    pub b: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct BoolAndU32 {
    pub a: bool,
    pub b: u32,
}

#[link(name = "test", kind = "static")]
extern "C" {
    fn many_args(
        a: *mut (),
        b: *mut (),
        c: *const i8,
        d: u64,
        e: bool,
        f: BoolAndU32,
        g: *mut (),
        h: TwoU64s,
        i: *mut (),
        j: *mut (),
        k: *mut (),
        l: *mut (),
        m: *const i8,
    ) -> i32;
}

fn main() {
    let two_u64s = TwoU64s { a: 1, b: 2 };
    let bool_and_u32 = BoolAndU32 { a: true, b: 3 };
    let string = CString::new("Hello world").unwrap();
    unsafe {
        many_args(
            null_mut(),
            null_mut(),
            null_mut(),
            4,
            true,
            bool_and_u32,
            null_mut(),
            two_u64s,
            null_mut(),
            null_mut(),
            null_mut(),
            null_mut(),
            string.as_ptr(),
        );
    }
}
