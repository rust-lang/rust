#![crate_type = "dylib"]
#![allow(dead_code)]

#[no_mangle] pub extern "C" fn fun1() {}
#[no_mangle] extern "C" fn fun2() {}

mod foo {
    #[no_mangle] pub extern "C" fn fun3() {}
}
pub mod bar {
    #[no_mangle] pub extern "C" fn fun4() {}
}

#[no_mangle] pub fn fun5() {}
