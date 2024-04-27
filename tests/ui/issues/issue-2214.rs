//@ run-pass
//@ ignore-wasm32 wasi-libc does not have lgamma
//@ ignore-sgx no libc

use std::ffi::{c_double, c_int};
use std::mem;

fn to_c_int(v: &mut isize) -> &mut c_int {
    unsafe { mem::transmute_copy(&v) }
}

fn lgamma(n: c_double, value: &mut isize) -> c_double {
    unsafe {
        return m::lgamma(n, to_c_int(value));
    }
}

mod m {
    use std::ffi::{c_double, c_int};
    extern "C" {
        #[cfg(all(unix, not(target_os = "vxworks")))]
        #[link_name="lgamma_r"]
        pub fn lgamma(n: c_double, sign: &mut c_int) -> c_double;
        #[cfg(windows)]
        #[link_name = "lgamma"]
        pub fn lgamma(n: c_double, sign: &mut c_int) -> c_double;
        #[cfg(target_os = "vxworks")]
        #[link_name = "lgamma"]
        pub fn lgamma(n: c_double, sign: &mut c_int) -> c_double;
    }
}

pub fn main() {
    let mut y: isize = 5;
    let x: &mut isize = &mut y;
    assert_eq!(lgamma(1.0 as c_double, x), 0.0 as c_double);
}
