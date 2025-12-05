//@ compile-flags: -O -Zmir-opt-level=2
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]
#![feature(step_trait)]

// EMIT_MIR checked_ops.step_forward.PreCodegen.after.mir
pub fn step_forward(x: u16, n: usize) -> u16 {
    // This uses `u16` so that the conversion to usize is always widening.

    // CHECK-LABEL: fn step_forward
    // CHECK: inlined{{.+}}forward
    std::iter::Step::forward(x, n)
}

// EMIT_MIR checked_ops.checked_shl.PreCodegen.after.mir
pub fn checked_shl(x: u32, rhs: u32) -> Option<u32> {
    // CHECK-LABEL: fn checked_shl
    // CHECK: [[TEMP:_[0-9]+]] = ShlUnchecked(copy _1, copy _2)
    // CHECK: _0 = Option::<u32>::Some({{move|copy}} [[TEMP]])
    x.checked_shl(rhs)
}

// EMIT_MIR checked_ops.use_checked_sub.PreCodegen.after.mir
pub fn use_checked_sub(x: u32, rhs: u32) {
    // We want this to be equivalent to open-coding it, leaving no `Option`s around.
    // FIXME(#138544): It's not yet.

    // CHECK-LABEL: fn use_checked_sub
    // CHECK: inlined{{.+}}u32{{.+}}checked_sub
    // CHECK: [[DELTA:_[0-9]+]] = SubUnchecked(copy _1, copy _2)
    // CHECK: [[TEMP1:_.+]] = Option::<u32>::Some(move [[DELTA]]);
    // CHECK: [[TEMP2:_.+]] = {{move|copy}} (([[TEMP1]] as Some).0: u32);
    // CHECK: do_something({{move|copy}} [[TEMP2]])
    if let Some(delta) = x.checked_sub(rhs) {
        do_something(delta);
    }
}

// EMIT_MIR checked_ops.saturating_sub_at_home.PreCodegen.after.mir
pub fn saturating_sub_at_home(lhs: u32, rhs: u32) -> u32 {
    // FIXME(#138544): Similarly here, the `Option` ought to optimize away

    // CHECK-LABEL: fn saturating_sub_at_home
    // CHECK: [[DELTA:_[0-9]+]] = SubUnchecked(copy _1, copy _2)
    // CHECK: [[TEMP1:_.+]] = Option::<u32>::Some({{move|copy}} [[DELTA]]);
    // CHECK: _0 = {{move|copy}} (([[TEMP1]] as Some).0: u32);
    u32::checked_sub(lhs, rhs).unwrap_or(0)
}

unsafe extern "Rust" {
    safe fn do_something(_: u32);
}
