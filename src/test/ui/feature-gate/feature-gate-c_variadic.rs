#![crate_type="lib"]

pub unsafe extern "C" fn test(_: i32, ap: ...) { }
//~^ C-varaidic functions are unstable
