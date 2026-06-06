//@ add-minicore
//@ only-wasm32
//@ assembly-output: emit-asm
//@ compile-flags: -C opt-level=2
//@ needs-llvm-components: webassembly
//@ min-llvm-version: 23

#![crate_type = "lib"]
#![no_std]
#![no_core]
#![feature(no_core, lang_items, f128, abi_wasm_multivalue)]

extern crate minicore;

// CHECK: .functype f1 () -> ()
#[no_mangle]
pub extern "C" fn f1() {}

// CHECK: .functype f1mv () -> ()
#[no_mangle]
pub extern "wasm-multivalue" fn f1mv() {}

// CHECK: .functype f2 (f32, f64) -> (i32)
#[no_mangle]
pub extern "C" fn f2(_a: f32, _b: f64) -> i32 {
    loop {}
}

// CHECK: .functype f2mv (f32, f64) -> (i32)
#[no_mangle]
pub extern "wasm-multivalue" fn f2mv(_a: f32, _b: f64) -> i32 {
    loop {}
}

// CHECK: .functype f3 (i32, i64, i64) -> ()
#[no_mangle]
pub extern "C" fn f3(_x: f128) -> i128 {
    loop {}
}

// CHECK: .functype f3mv (i64, i64) -> (i64, i64)
#[no_mangle]
pub extern "wasm-multivalue" fn f3mv(_x: f128) -> i128 {
    loop {}
}

#[repr(C)]
pub struct Foo4 {}

#[repr(C)]
pub union Bar4 {
    _empty: (),
}

// CHECK: .functype f4 () -> ()
#[no_mangle]
pub extern "C" fn f4(_x: Foo4) -> Bar4 {
    Bar4 { _empty: () }
}

// CHECK: .functype f4mv () -> ()
#[no_mangle]
pub extern "wasm-multivalue" fn f4mv(_x: Foo4) -> Bar4 {
    Bar4 { _empty: () }
}

#[repr(C)]
pub struct Foo5 {
    a: i32,
}

#[repr(C)]
pub union Bar5 {
    a: i32,
}

// CHECK: .functype f5 (i32) -> (i32)
#[no_mangle]
pub extern "C" fn f5(x: Foo5) -> Bar5 {
    Bar5 { a: x.a }
}

// CHECK: .functype f5mv (i32) -> (i32)
#[no_mangle]
pub extern "wasm-multivalue" fn f5mv(x: Foo5) -> Bar5 {
    Bar5 { a: x.a }
}

#[repr(C)]
pub struct Foo6 {
    a: i32,
    b: i32,
}

// CHECK: .functype f6 (i32, i32) -> ()
#[no_mangle]
pub extern "C" fn f6(x: Foo6) -> Foo6 {
    x
}

// CHECK: .functype f6mv (i32, i32) -> (i32, i32)
#[no_mangle]
pub extern "wasm-multivalue" fn f6mv(x: Foo6) -> Foo6 {
    x
}
