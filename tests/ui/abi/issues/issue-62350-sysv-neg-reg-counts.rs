// run-pass
#![allow(dead_code)]
#![allow(improper_ctypes)]

// ignore-wasm32-bare no libc to test ffi with

#[derive(Copy, Clone)]
pub struct QuadFloats {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
}

mod rustrt {
    use super::QuadFloats;

    #[link(name = "rust_test_helpers", kind = "static")]
    extern "C" {
        pub fn get_c_exhaust_sysv64_ints(
            _: *const (),
            _: *const (),
            _: *const (),
            _: *const (),
            _: *const (),
            _: *const (),
            _: *const (),
            h: QuadFloats,
        ) -> f32;
    }
}

fn test() {
    unsafe {
        let null = std::ptr::null();
        let q = QuadFloats { a: 10.2, b: 20.3, c: 30.4, d: 40.5 };
        assert_eq!(
            rustrt::get_c_exhaust_sysv64_ints(null, null, null, null, null, null, null, q),
            q.c,
        );
    }
}

pub fn main() {
    test();
}
