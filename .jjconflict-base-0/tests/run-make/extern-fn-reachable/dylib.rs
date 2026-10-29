#![crate_type = "dylib"]
#![allow(dead_code)]

// `pub` extern fn here is a Rust nameres visibility concept, and should not affect symbol
// visibility in the dylib.
#[no_mangle]
pub extern "C" fn fun1() {}

// (Lack of) `pub` for the extern fn here is a Rust nameres visibility concept, and should not
// affect symbol visibility in the dylib.
#[no_mangle]
extern "C" fn fun2() {}

// Modules are a Rust nameres concept, and should not affect symbol visibility in the dylib if the
// extern fn is nested inside a module.
mod foo {
    #[no_mangle]
    pub extern "C" fn fun3() {}
}

// Similarly, the Rust visibility of the containing module is a Rust nameres concept, and should not
// affect symbol visibility in the dylib.
pub mod bar {
    #[no_mangle]
    pub extern "C" fn fun4() {}
}

// Non-extern `#[no_mangle]` fn should induce a symbol visible in the dylib.
#[no_mangle]
pub fn fun5() {}

// The Rust visibility of the fn should not affect is symbol visibility in the dylib.
#[no_mangle]
fn fun6() {}
