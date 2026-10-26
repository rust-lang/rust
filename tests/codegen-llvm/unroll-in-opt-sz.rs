//@ revisions: O2 S Z
//@[O2] compile-flags: -Copt-level=2
//@[S] compile-flags: -Copt-level=s
//@[Z] compile-flags: -Copt-level=z

// `-Copt-level=s`/`z` pass `PTO.LoopUnrolling = false`, so LLVM's unroller runs `OnlyWhenForced`.
// The O2 side is asserted too, so disabling unrolling everywhere cannot pass. Calls are counted
// because this knob emits no `llvm.loop.unroll.*` metadata.

#![crate_type = "lib"]

unsafe extern "C" {
    fn side_effect(i: u32);
}

// O2-LABEL: @four_iters(
// O2-COUNT-4: tail call void @side_effect
// S-LABEL: @four_iters(
// S: phi i32
// S: tail call void @side_effect
// S-NOT: tail call void @side_effect
// Z-LABEL: @four_iters(
// Z: phi i32
// Z: tail call void @side_effect
// Z-NOT: tail call void @side_effect
#[no_mangle]
pub fn four_iters() {
    for i in 0..4u32 {
        unsafe { side_effect(i) }
    }
}
