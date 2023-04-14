// Check that `UnwindAction::Unreachable` is not generated for unwindable intrinsics.
// ignore-wasm32 compiled with panic=abort by default
#![feature(core_intrinsics)]

// EMIT_MIR issue_104451_unwindable_intrinsics.main.AbortUnwindingCalls.after.mir
fn main() {
    unsafe {
        core::intrinsics::const_eval_select((), ow_ct, ow_ct)
    }
}

const fn ow_ct() -> ! {
    panic!();
}
