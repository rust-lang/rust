//@ revisions: wasm32-unknown wasm64-unknown wasm32-wasip1
//@ add-core-stubs
//@ assembly-output: emit-asm
//@ [wasm32-unknown] compile-flags: --target wasm32-unknown-unknown
//@ [wasm64-unknown] compile-flags: --target wasm64-unknown-unknown
//@ [wasm32-wasip1] compile-flags: --target wasm32-wasip1
//@ [wasm32-unknown] needs-llvm-components: webassembly
//@ [wasm64-unknown] needs-llvm-components: webassembly
//@ [wasm32-wasip1] needs-llvm-components: webassembly

#![crate_type = "lib"]
#![feature(no_core, asm_experimental_arch, f128, linkage, fn_align)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK: .section  .text.nop,"",@
// CHECK: .globl nop
// CHECK-LABEL: nop:
// CHECK: .functype nop () -> ()
// CHECK-NOT: .size
// CHECK: end_function
// CHECK-LABEL: .Lfunc_end_nop:
#[no_mangle]
#[unsafe(naked)]
extern "C" fn nop() {
    naked_asm!("nop")
}

// CHECK: .section  .text.weak_nop,"",@
// CHECK: .weak weak_nop
// CHECK-LABEL: nop:
// CHECK: .functype weak_nop () -> ()
// CHECK-NOT: .size
// CHECK: end_function
#[no_mangle]
#[unsafe(naked)]
#[linkage = "weak"]
extern "C" fn weak_nop() {
    naked_asm!("nop")
}

// CHECK-LABEL: fn_i8_i8:
// CHECK-NEXT: .functype fn_i8_i8 (i32) -> (i32)
//
// CHECK-NEXT: local.get 0
// CHECK-NEXT: local.get 0
// CHECK-NEXT: i32.mul
//
// CHECK-NEXT: end_function
#[no_mangle]
#[unsafe(naked)]
extern "C" fn fn_i8_i8(num: i8) -> i8 {
    naked_asm!("local.get 0", "local.get 0", "i32.mul")
}

// CHECK-LABEL: fn_i8_i8_i8:
// CHECK: .functype fn_i8_i8_i8 (i32, i32) -> (i32)
#[no_mangle]
#[unsafe(naked)]
extern "C" fn fn_i8_i8_i8(a: i8, b: i8) -> i8 {
    naked_asm!("local.get 1", "local.get 0", "i32.mul")
}

// CHECK-LABEL: fn_unit_i8:
// CHECK: .functype fn_unit_i8 () -> (i32)
#[no_mangle]
#[unsafe(naked)]
extern "C" fn fn_unit_i8() -> i8 {
    naked_asm!("i32.const 42")
}

// CHECK-LABEL: fn_i8_unit:
// CHECK: .functype fn_i8_unit (i32) -> ()
#[no_mangle]
#[unsafe(naked)]
extern "C" fn fn_i8_unit(_: i8) {
    naked_asm!("nop")
}

// CHECK-LABEL: fn_i32_i32:
// CHECK: .functype fn_i32_i32 (i32) -> (i32)
#[no_mangle]
#[unsafe(naked)]
extern "C" fn fn_i32_i32(num: i32) -> i32 {
    naked_asm!("local.get 0", "local.get 0", "i32.mul")
}

// CHECK-LABEL: fn_i64_i64:
// CHECK: .functype fn_i64_i64 (i64) -> (i64)
#[no_mangle]
#[unsafe(naked)]
extern "C" fn fn_i64_i64(num: i64) -> i64 {
    naked_asm!("local.get 0", "local.get 0", "i64.mul")
}

// CHECK-LABEL: fn_i128_i128:
// wasm32-unknown: .functype fn_i128_i128 (i32, i64, i64) -> ()
// wasm32-wasip1: .functype fn_i128_i128 (i32, i64, i64) -> ()
// wasm64-unknown: .functype fn_i128_i128 (i64, i64, i64) -> ()
#[allow(improper_ctypes_definitions)]
#[no_mangle]
#[unsafe(naked)]
extern "C" fn fn_i128_i128(num: i128) -> i128 {
    naked_asm!(
        "local.get       0",
        "local.get       2",
        "i64.store       8",
        "local.get       0",
        "local.get       1",
        "i64.store       0",
    )
}

// CHECK-LABEL: fn_f128_f128:
// wasm32-unknown: .functype fn_f128_f128 (i32, i64, i64) -> ()
// wasm32-wasip1: .functype fn_f128_f128 (i32, i64, i64) -> ()
// wasm64-unknown: .functype fn_f128_f128 (i64, i64, i64) -> ()
#[no_mangle]
#[unsafe(naked)]
extern "C" fn fn_f128_f128(num: f128) -> f128 {
    naked_asm!(
        "local.get       0",
        "local.get       2",
        "i64.store       8",
        "local.get       0",
        "local.get       1",
        "i64.store       0",
    )
}

#[repr(C)]
struct Compound {
    a: u16,
    b: i64,
}

// CHECK-LABEL: fn_compound_compound:
// wasm32-unknown: .functype fn_compound_compound (i32, i32) -> ()
// wasm32-wasip1: .functype fn_compound_compound (i32, i32) -> ()
// wasm64-unknown: .functype fn_compound_compound (i64, i64) -> ()
#[no_mangle]
#[unsafe(naked)]
extern "C" fn fn_compound_compound(_: Compound) -> Compound {
    // this is the wasm32-wasip1 assembly
    naked_asm!(
        "local.get       0",
        "local.get       1",
        "i64.load        8",
        "i64.store       8",
        "local.get       0",
        "local.get       1",
        "i32.load16_u    0",
        "i32.store16     0",
    )
}

#[repr(C)]
struct WrapperI32(i32);

// CHECK-LABEL: fn_wrapperi32_wrapperi32:
// CHECK: .functype fn_wrapperi32_wrapperi32 (i32) -> (i32)
#[no_mangle]
#[unsafe(naked)]
extern "C" fn fn_wrapperi32_wrapperi32(_: WrapperI32) -> WrapperI32 {
    naked_asm!("local.get       0")
}

#[repr(C)]
struct WrapperI64(i64);

// CHECK-LABEL: fn_wrapperi64_wrapperi64:
// CHECK: .functype fn_wrapperi64_wrapperi64 (i64) -> (i64)
#[no_mangle]
#[unsafe(naked)]
extern "C" fn fn_wrapperi64_wrapperi64(_: WrapperI64) -> WrapperI64 {
    naked_asm!("local.get       0")
}

#[repr(C)]
struct WrapperF32(f32);

// CHECK-LABEL: fn_wrapperf32_wrapperf32:
// CHECK: .functype fn_wrapperf32_wrapperf32 (f32) -> (f32)
#[no_mangle]
#[unsafe(naked)]
extern "C" fn fn_wrapperf32_wrapperf32(_: WrapperF32) -> WrapperF32 {
    naked_asm!("local.get       0")
}

#[repr(C)]
struct WrapperF64(f64);

// CHECK-LABEL: fn_wrapperf64_wrapperf64:
// CHECK: .functype fn_wrapperf64_wrapperf64 (f64) -> (f64)
#[no_mangle]
#[unsafe(naked)]
extern "C" fn fn_wrapperf64_wrapperf64(_: WrapperF64) -> WrapperF64 {
    naked_asm!("local.get       0")
}
