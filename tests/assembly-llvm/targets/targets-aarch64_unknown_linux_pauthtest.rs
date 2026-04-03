//@ add-minicore
//@ assembly-output: emit-asm
//@ only-aarch64-unknown-linux-pauthtest
//@ revisions: aarch64_unknown_linux_pauthtest
//@ [aarch64_unknown_linux_pauthtest] compile-flags: --target=aarch64-unknown-linux-pauthtest
//@ [aarch64_unknown_linux_pauthtest] needs-llvm-components: aarch64

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;

#[no_mangle]
#[inline(never)]
pub extern "C" fn c_func(a: i32) -> i32 {
    a
}

#[no_mangle]
#[inline(never)]
fn call_through(f: extern "C" fn(i32) -> i32, x: i32) -> i32 {
    f(x)
}

#[no_mangle]
#[inline(never)]
pub fn call_c_func(x: i32) -> i32 {
    call_through(c_func, x)
}

// CHECK-LABEL: call_through:
// CHECK:       mov     [[PTR:x[0-9]+]], x0
// CHECK:       mov     w0, w1
// CHECK:       braaz   [[PTR]]

// CHECK-LABEL: call_c_func:
// CHECK:       adrp    [[GOT_REG:x[0-9]+]], :got:c_func
// CHECK:       ldr     [[GOT_REG]], [[[GOT_REG]], :got_lo12:c_func]
// CHECK:       paciza  [[FN_REG:x[0-9]+]]
// CHECK:       mov     w1, w0
// CHECK:       mov     x0, [[FN_REG]]
// CHECK:       b       call_through
