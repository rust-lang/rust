#![feature(raw_dylib_elf)]
#![allow(incomplete_features)]

#[link(name = "library", kind = "raw-dylib")]
unsafe extern "C" {
    safe fn this_is_a_library_function() -> core::ffi::c_int;
}

fn main() {
    println!("{}", this_is_a_library_function())
}
