//@ assembly-output: emit-asm
//@ only-aarch64-unknown-linux-pauthtest
//@ revisions: aarch64_unknown_linux_pauthtest
//@ [aarch64_unknown_linux_pauthtest] compile-flags: --target=aarch64-unknown-linux-pauthtest
//@ [aarch64_unknown_linux_pauthtest] needs-llvm-components: aarch64

#![no_std]
#![crate_type = "lib"]

#[no_mangle]
#[inline(never)]
pub extern "C" fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[no_mangle]
#[inline(never)]
fn call_through(f: extern "C" fn(i32, i32) -> i32, x: i32) -> i32 {
    f(x, 1)
}

#[no_mangle]
#[inline(never)]
pub fn call_add(x: i32) -> i32 {
    call_through(add, x)
}

// CHECK-LABEL: call_through:
// CHECK:       mov     [[PTR:x[0-9]+]], x0
// CHECK:       mov     w0, w1
// CHECK:       mov     w1, #1
// CHECK:       braaz   [[PTR]]

// CHECK-LABEL: call_add:
// CHECK:       adrp    [[GOT_REG:x[0-9]+]], :got:add
// CHECK:       ldr     [[GOT_REG]], [[[GOT_REG]], :got_lo12:add]
// CHECK:       paciza  [[FN_REG:x[0-9]+]]
// CHECK:       mov     w1, w0
// CHECK:       mov     x0, [[FN_REG]]
// CHECK:       b       call_through
