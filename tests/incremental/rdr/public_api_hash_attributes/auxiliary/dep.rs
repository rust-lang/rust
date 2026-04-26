//@ compile-flags: -Z public-api-hash

#![crate_name = "dep"]
#![crate_type = "rlib"]

#[cfg(any(cpass1, cpass2))]
pub fn generic<T: std::fmt::Debug>(v: T) {
    panic!("{v:?}");
}

#[cfg(any(cpass3, bpass4, bfail5))]
pub fn generic<T: std::fmt::Debug>(v: T) {
    panic!("{v:?}");
}
