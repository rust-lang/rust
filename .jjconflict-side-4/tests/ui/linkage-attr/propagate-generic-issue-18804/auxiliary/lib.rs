#![crate_type = "rlib"]
#![feature(linkage)]

pub fn foo<T>() -> *const () {
    extern "C" {
        #[linkage = "extern_weak"]
        static FOO: *const ();
    }
    unsafe { FOO }
}
