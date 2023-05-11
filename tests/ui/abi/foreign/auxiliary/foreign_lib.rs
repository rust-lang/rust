#![crate_name = "foreign_lib"]
#![feature(rustc_private)]

pub mod rustrt {
    extern crate libc;

    #[link(name = "rust_test_helpers", kind = "static")]
    extern "C" {
        pub fn rust_get_test_int() -> libc::intptr_t;
    }
}

pub mod rustrt2 {
    extern crate libc;

    extern "C" {
        pub fn rust_get_test_int() -> libc::intptr_t;
    }
}

pub mod rustrt3 {
    // Different type, but same ABI (on all supported platforms).
    // Ensures that we don't ICE or trigger LLVM asserts when
    // importing the same symbol under different types.
    // See https://github.com/rust-lang/rust/issues/32740.
    extern "C" {
        pub fn rust_get_test_int() -> *const u8;
    }
}

pub fn local_uses() {
    unsafe {
        let x = rustrt::rust_get_test_int();
        assert_eq!(x, rustrt2::rust_get_test_int());
        assert_eq!(x as *const _, rustrt3::rust_get_test_int());
    }
}
