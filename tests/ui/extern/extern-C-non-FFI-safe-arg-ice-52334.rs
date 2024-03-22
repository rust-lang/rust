// test for ICE when casting extern "C" fn when it has a non-FFI-safe argument
// issue: rust-lang/rust#52334
//@ check-pass
//@ normalize-stderr-test "\[i8\]" -> "[i8 or u8 (arch dependant)]"
//@ normalize-stderr-test "\[u8\]" -> "[i8 or u8 (arch dependant)]"

type Foo = extern "C" fn(::std::ffi::CStr);
//~^ WARN `extern` fn uses type `[i8]`, which is not FFI-safe
extern "C" {
    fn meh(blah: Foo);
    //~^ WARN `extern` block uses type `[i8]`, which is not FFI-safe
}

fn main() {
    meh as usize;
}
