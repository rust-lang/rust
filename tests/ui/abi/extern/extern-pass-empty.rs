//@ run-pass
#![allow(improper_ctypes)] // FIXME: this test is inherently not FFI-safe.

// Test a foreign function that accepts empty struct.

//@ ignore-msvc
//@ ignore-emscripten emcc asserts on an empty struct as an argument

#[repr(C)]
struct TwoU8s {
    one: u8,
    two: u8,
}

#[repr(C)]
struct ManyInts {
    arg1: i8,
    arg2: i16,
    arg3: i32,
    arg4: i16,
    arg5: i8,
    arg6: TwoU8s,
}

#[repr(C)]
struct Empty;

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    fn rust_dbg_extern_empty_struct(v1: ManyInts, e: Empty, v2: ManyInts);
}

pub fn main() {
    unsafe {
        let x = ManyInts {
            arg1: 2,
            arg2: 3,
            arg3: 4,
            arg4: 5,
            arg5: 6,
            arg6: TwoU8s { one: 7, two: 8 },
        };
        let y = ManyInts {
            arg1: 1,
            arg2: 2,
            arg3: 3,
            arg4: 4,
            arg5: 5,
            arg6: TwoU8s { one: 6, two: 7 },
        };
        let empty = Empty;
        rust_dbg_extern_empty_struct(x, empty, y);
    }
}
