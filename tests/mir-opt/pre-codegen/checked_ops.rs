// skip-filecheck
//@ compile-flags: -O -Zmir-opt-level=2
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]
#![feature(step_trait)]

// EMIT_MIR checked_ops.step_forward.PreCodegen.after.mir
pub fn step_forward(x: u16, n: usize) -> u16 {
    // This uses `u16` so that the conversion to usize is always widening.
    std::iter::Step::forward(x, n)
}

// EMIT_MIR checked_ops.checked_shl.PreCodegen.after.mir
pub fn checked_shl(x: u32, rhs: u32) -> Option<u32> {
    x.checked_shl(rhs)
}
