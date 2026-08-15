//! Tests that unwinding from an asm block is caught and forced to abort
//! when `-C panic=abort`.

//@ compile-flags: -C panic=abort
//@ needs-asm-support

#![feature(asm_unwind)]

// EMIT_MIR asm_unwind_panic_abort.main.AbortUnwindingCalls.after.mir
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: asm!(
    // CHECK-SAME: unwind: [[unwind:bb.*]]]
    // CHECK: [[unwind]] (cleanup)
    // CHECK-NEXT: terminate(abi)
    unsafe {
        std::arch::asm!("", options(may_unwind));
    }
}
