// compile-flags: -O -Zmir-opt-level=2 -Cdebuginfo=2
// needs-unwind
// ignore-debug
// only-x86_64

#![crate_type = "lib"]
#![feature(step_trait)]

// EMIT_MIR checked_ops.step_forward.PreCodegen.after.mir
pub fn step_forward(x: u32, n: usize) -> u32 {
    std::iter::Step::forward(x, n)
}

// EMIT_MIR checked_ops.checked_shl.PreCodegen.after.mir
pub fn checked_shl(x: u32, rhs: u32) -> Option<u32> {
    x.checked_shl(rhs)
}
