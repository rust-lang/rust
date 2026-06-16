//@ compile-flags: -Zmir-opt-level=0
// Non-coroutine unwind paths use BackwardIncompatibleDropHint where a future
// edition would insert StorageDead.

// EMIT_MIR unwind_drop_hint.test.built.after.mir

struct Droppable;

impl Drop for Droppable {
    fn drop(&mut self) {}
}

fn test() {
    // CHECK-LABEL: fn test(
    // The built MIR dump asserts the exact BackwardIncompatibleDropHint placement.
    // FileCheck runs on post-borrowck MIR, where CleanupPostBorrowck has removed
    // those hints again.
    // CHECK-NOT: BackwardIncompatibleDropHint
    let x = 42i32;
    let y = Droppable;
    may_panic();
}

fn may_panic() {}

fn main() {
    test();
}
