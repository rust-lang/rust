#![feature(raw_dylib_elf)]
#![allow(incomplete_features)]

#[link(name = "liblibrary.so.1", kind = "raw-dylib", modifiers = "+verbatim")]
unsafe extern "C" {
    safe fn this_is_a_library_function() -> core::ffi::c_int;
}

fn main() {
    println!("{}", this_is_a_library_function())
}
