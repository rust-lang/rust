#![feature(raw_dylib_elf)]
#![feature(thread_local)]
#![allow(incomplete_features)]

use std::ffi::c_int;

#[link(name = "library", kind = "raw-dylib")]
unsafe extern "C" {
    safe fn this_is_a_library_function() -> c_int;
    static mut global_variable: c_int;
    #[thread_local]
    static mut tls_variable: c_int;
}

fn main() {
    unsafe {
        println!("{} {} {}", this_is_a_library_function(), global_variable, tls_variable);
    }
}
