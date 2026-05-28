//@compile-flags: --crate-name result_large_err
#![warn(clippy::result_large_err)]
#![allow(clippy::large_enum_variant)]

fn f() -> Result<(), [u8; 511]> {
    todo!()
}
fn f2() -> Result<(), [u8; 512]> {
    //~^ ERROR: the `Err`-variant returned from this function is very large
    todo!()
}

struct IgnoredError {
    inner: [u8; 512],
}

fn f3() -> Result<(), IgnoredError> {
    todo!()
}

enum IgnoredErrorEnum {
    V1,
    V2 { inner: [u8; 512] },
}

fn f4() -> Result<(), IgnoredErrorEnum> {
    todo!()
}

fn main() {}
