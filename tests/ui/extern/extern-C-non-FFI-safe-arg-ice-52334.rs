// test for ICE when casting extern "C" fn when it has a non-FFI-safe argument
// issue: rust-lang/rust#52334
//@ check-pass
//@ normalize-stderr: "\[i8\]" -> "[i8 or u8 (arch dependant)]"
//@ normalize-stderr: "\[u8\]" -> "[i8 or u8 (arch dependant)]"

#![allow(function_casts_as_integer)]

type Foo = extern "C" fn(::std::ffi::CStr);
//~^ WARN `extern` callback uses type
extern "C" {
    fn meh(blah: Foo);
}

fn main() {
    meh as usize;
}
