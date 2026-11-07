//! Issue #50021

#![crate_type = "cdylib"]

mod m1 {
    #[link(wasm_import_module = "m1")]
    extern "C" {
        pub fn f();
    }
    #[link(wasm_import_module = "m1")]
    extern "C" {
        pub fn g();
    }
}

mod m2 {
    #[link(wasm_import_module = "m2")]
    extern "C" {
        pub fn f(_: i32);
    }
}

#[no_mangle]
pub unsafe fn run() {
    m1::f();
    m1::g();

    // In generated code, expected:
    // (import "m2" "f" (func $f (param i32)))
    // but got:
    // (import "m1" "f" (func $f (param i32)))
    m2::f(0);
}
