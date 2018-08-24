#![feature(rustc_attrs)]
#![deny(non_snake_case)]

#[no_mangle]
pub extern "C" fn SparklingGenerationForeignFunctionInterface() {}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
