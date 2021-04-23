#![crate_type = "cdylib"]
#![deny(warnings)]
#![feature(wasm_abi)]

#[repr(C)]
#[derive(PartialEq, Debug)]
pub struct TwoI32 {
    pub a: i32,
    pub b: i32,
}

#[no_mangle]
pub extern "wasm" fn return_two_i32() -> TwoI32 {
    TwoI32 { a: 1, b: 2 }
}

#[repr(C)]
#[derive(PartialEq, Debug)]
pub struct TwoI64 {
    pub a: i64,
    pub b: i64,
}

#[no_mangle]
pub extern "wasm" fn return_two_i64() -> TwoI64 {
    TwoI64 { a: 3, b: 4 }
}

#[repr(C)]
#[derive(PartialEq, Debug)]
pub struct TwoF32 {
    pub a: f32,
    pub b: f32,
}

#[no_mangle]
pub extern "wasm" fn return_two_f32() -> TwoF32 {
    TwoF32 { a: 5., b: 6. }
}

#[repr(C)]
#[derive(PartialEq, Debug)]
pub struct TwoF64 {
    pub a: f64,
    pub b: f64,
}

#[no_mangle]
pub extern "wasm" fn return_two_f64() -> TwoF64 {
    TwoF64 { a: 7., b: 8. }
}

#[repr(C)]
#[derive(PartialEq, Debug)]
pub struct Mishmash {
    pub a: f64,
    pub b: f32,
    pub c: i32,
    pub d: i64,
    pub e: TwoI32,
}

#[no_mangle]
pub extern "wasm" fn return_mishmash() -> Mishmash {
    Mishmash { a: 9., b: 10., c: 11, d: 12, e: TwoI32 { a: 13, b: 14 } }
}

#[link(wasm_import_module = "host")]
extern "wasm" {
    fn two_i32() -> TwoI32;
    fn two_i64() -> TwoI64;
    fn two_f32() -> TwoF32;
    fn two_f64() -> TwoF64;
    fn mishmash() -> Mishmash;
}

#[no_mangle]
pub unsafe extern "C" fn call_imports() {
    assert_eq!(two_i32(), TwoI32 { a: 100, b: 101 });
    assert_eq!(two_i64(), TwoI64 { a: 102, b: 103 });
    assert_eq!(two_f32(), TwoF32 { a: 104., b: 105. });
    assert_eq!(two_f64(), TwoF64 { a: 106., b: 107. });
    assert_eq!(
        mishmash(),
        Mishmash { a: 108., b: 109., c: 110, d: 111, e: TwoI32 { a: 112, b: 113 } }
    );
}
