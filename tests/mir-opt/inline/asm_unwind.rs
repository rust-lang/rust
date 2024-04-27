// Tests inlining of `may_unwind` inline assembly.
//
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ needs-asm-support
//@ needs-unwind
//@ compile-flags: -Zinline-mir-hint-threshold=1000 -C debuginfo=full
#![feature(asm_unwind)]

struct D;

impl Drop for D {
    fn drop(&mut self) {}
}

#[inline]
fn foo() {
    let _d = D;
    unsafe { std::arch::asm!("", options(may_unwind)) };
}

// EMIT_MIR asm_unwind.main.Inline.diff
pub fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: (inlined foo)
    // CHECK: asm!("", options(MAY_UNWIND)) -> [return: {{bb.*}}, unwind: [[unwind:bb.*]]];
    // CHECK: [[unwind]] (cleanup)
    foo();
}
