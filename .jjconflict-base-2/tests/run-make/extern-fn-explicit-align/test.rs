// Issue #80127: Passing structs via FFI should work with explicit alignment.

use std::ffi::{CStr, c_char};
use std::ptr::null_mut;

#[repr(C)]
pub struct BoolAndU32 {
    pub a: bool,
    pub b: u32,
}

#[repr(C)]
#[repr(align(16))]
pub struct TwoU64s {
    pub a: u64,
    pub b: u64,
}

#[repr(C)]
pub struct WrappedU64s {
    pub a: TwoU64s,
}

#[repr(C)]
// Even though requesting align 1 can never change the alignment, it still affects the ABI
// on some platforms like i686-windows.
#[repr(align(1))]
pub struct LowerAlign {
    pub a: u64,
    pub b: u64,
}

#[repr(C)]
#[repr(packed)]
pub struct Packed {
    pub a: u64,
    pub b: u64,
}

#[link(name = "test", kind = "static")]
extern "C" {
    fn many_args(
        a: *mut (),
        b: *mut (),
        c: *const c_char,
        d: u64,
        e: bool,
        f: BoolAndU32,
        g: *mut (),
        h: TwoU64s,
        i: *mut (),
        j: WrappedU64s,
        k: *mut (),
        l: LowerAlign,
        m: *mut (),
        n: Packed,
        o: *const c_char,
    ) -> i32;
}

const STRING: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"Hello world\0") };

fn main() {
    let bool_and_u32 = BoolAndU32 { a: true, b: 1337 };
    let two_u64s = TwoU64s { a: 1, b: 2 };
    let wrapped = WrappedU64s { a: TwoU64s { a: 3, b: 4 } };
    let lower = LowerAlign { a: 5, b: 6 };
    let packed = Packed { a: 7, b: 8 };
    let string = STRING;
    unsafe {
        many_args(
            null_mut(),
            null_mut(),
            null_mut(),
            42,
            true,
            bool_and_u32,
            null_mut(),
            two_u64s,
            null_mut(),
            wrapped,
            null_mut(),
            lower,
            null_mut(),
            packed,
            string.as_ptr(),
        );
    }
}
