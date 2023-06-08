// Tests inlining of `may_unwind` inline assembly.
//
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// needs-asm-support
#![feature(asm_unwind)]

struct D;

impl Drop for D {
    fn drop(&mut self) {}
}

#[inline(always)]
fn foo() {
    let _d = D;
    unsafe { std::arch::asm!("", options(may_unwind)) };
}

// EMIT_MIR asm_unwind.main.Inline.diff
pub fn main() {
    foo();
}
