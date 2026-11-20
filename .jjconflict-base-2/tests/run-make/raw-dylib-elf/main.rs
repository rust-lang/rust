#![feature(raw_dylib_elf)]
#![feature(thread_local)]
#![allow(incomplete_features)]

use std::ffi::{CStr, c_char, c_int, c_long};

#[link(name = "library", kind = "raw-dylib")]
unsafe extern "C" {
    safe fn this_is_a_library_function() -> c_int;
    static mut global_variable: c_int;
    #[thread_local]
    static mut tls_variable: c_int;
    safe static const_array: [c_long; 2];
    safe static const_string: *const c_char;
}

fn main() {
    unsafe {
        println!(
            "{} {} {} {} {} {}",
            this_is_a_library_function(),
            global_variable,
            tls_variable,
            const_array[0],
            const_array[1],
            CStr::from_ptr(const_string).to_str().unwrap(),
        );
    }
}
