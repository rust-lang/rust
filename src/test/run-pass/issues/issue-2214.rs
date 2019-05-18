// run-pass
// ignore-wasm32-bare no libc to test ffi with
// ignore-sgx no libc
#![feature(rustc_private)]

extern crate libc;

use std::mem;
use libc::{c_double, c_int};

fn to_c_int(v: &mut isize) -> &mut c_int {
    unsafe {
        mem::transmute_copy(&v)
    }
}

fn lgamma(n: c_double, value: &mut isize) -> c_double {
    unsafe {
        return m::lgamma(n, to_c_int(value));
    }
}

mod m {
    use libc::{c_double, c_int};

    #[link_name = "m"]
    extern {
        #[cfg(any(unix, target_os = "cloudabi"))]
        #[link_name="lgamma_r"]
        pub fn lgamma(n: c_double, sign: &mut c_int) -> c_double;
        #[cfg(windows)]
        #[link_name="lgamma"]
        pub fn lgamma(n: c_double, sign: &mut c_int) -> c_double;
    }
}

pub fn main() {
  let mut y: isize = 5;
  let x: &mut isize = &mut y;
  assert_eq!(lgamma(1.0 as c_double, x), 0.0 as c_double);
}
