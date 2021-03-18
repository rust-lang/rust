#![crate_type="lib"]

pub unsafe extern "C" fn test(_: i32, ap: ...) { }
//~^ C-variadic functions are unstable
