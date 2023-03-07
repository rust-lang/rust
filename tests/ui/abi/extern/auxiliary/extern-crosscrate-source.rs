#![crate_name = "externcallback"]
#![crate_type = "lib"]
#![feature(rustc_private)]

extern crate libc;

pub mod rustrt {
    extern crate libc;

    #[link(name = "rust_test_helpers", kind = "static")]
    extern "C" {
        pub fn rust_dbg_call(
            cb: extern "C" fn(libc::uintptr_t) -> libc::uintptr_t,
            data: libc::uintptr_t,
        ) -> libc::uintptr_t;
    }
}

pub fn fact(n: libc::uintptr_t) -> libc::uintptr_t {
    unsafe {
        println!("n = {}", n);
        rustrt::rust_dbg_call(cb, n)
    }
}

pub extern "C" fn cb(data: libc::uintptr_t) -> libc::uintptr_t {
    if data == 1 { data } else { fact(data - 1) * data }
}
