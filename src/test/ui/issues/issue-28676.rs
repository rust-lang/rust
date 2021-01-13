// run-pass
#![allow(dead_code)]
#![allow(improper_ctypes)]

// ignore-wasm32-bare no libc to test ffi with

#[derive(Copy, Clone)]
pub struct Quad {
    a: u64,
    b: u64,
    c: u64,
    d: u64,
}

mod rustrt {
    use super::Quad;

    #[link(name = "rust_test_helpers", kind = "static")]
    extern "C" {
        pub fn get_c_many_params(
            _: *const (),
            _: *const (),
            _: *const (),
            _: *const (),
            f: Quad,
        ) -> u64;
    }
}

fn test() {
    unsafe {
        let null = std::ptr::null();
        let q = Quad { a: 1, b: 2, c: 3, d: 4 };
        assert_eq!(rustrt::get_c_many_params(null, null, null, null, q), q.c);
    }
}

pub fn main() {
    test();
}
