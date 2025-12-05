#![deny(improper_ctypes_definitions)]

extern "C" fn foo<T: ?Sized + 'static>() -> Option<&'static T> {
    //~^ ERROR `extern` fn uses type `Option<&T>`, which is not FFI-safe
    None
}

fn main() {}
