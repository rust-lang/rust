// test for #52334 ICE when casting extern "C" fn when it has a non-FFI-safe argument
//@ check-pass

type Foo = extern "C" fn(::std::ffi::CStr);
//~^ WARN `extern` fn uses type `[i8]`, which is not FFI-safe
extern "C" {
    fn meh(blah: Foo);
    //~^ WARN `extern` block uses type `[i8]`, which is not FFI-safe
}

fn main() {
    meh as usize;
}
