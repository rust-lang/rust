//! Tests that unwinding from an asm block is caught and forced to abort
//! when `-C panic=abort`.

// only-x86_64
// compile-flags: -C panic=abort
// no-prefer-dynamic

#![feature(asm_unwind)]

// EMIT_MIR asm_unwind_panic_abort.main.AbortUnwindingCalls.after.mir
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: asm!(
    // CHECK-SAME: unwind terminate(abi)
    unsafe {
        std::arch::asm!("", options(may_unwind));
    }
}
