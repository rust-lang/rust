//@ run-pass

#![allow(dead_code)]
#![feature(rustc_private)]

mod bar {
    extern "C" {}
}

mod zed {
    extern "C" {}
}

#[cfg(not(windows))]
mod mlibc {
    extern crate libc;
    use self::libc::{c_int, c_void, size_t, ssize_t};

    extern "C" {
        pub fn write(fd: c_int, buf: *const c_void, count: size_t) -> ssize_t;
    }
}

#[cfg(windows)]
mod mlibc {
    #![allow(non_snake_case)]

    use std::ffi::c_void;

    pub type BOOL = i32;
    pub type HANDLE = *mut c_void;

    #[link(name = "ntdll")]
    extern "system" {
        pub fn WriteFile(
            hfile: HANDLE,
            lpbuffer: *const u8,
            nnumberofbytestowrite: u32,
            lpnumberofbyteswritten: *mut u32,
            lpoverlapped: *mut c_void,
        ) -> BOOL;
    }
}

mod baz {
    extern "C" {}
}

pub fn main() {}
