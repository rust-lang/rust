//@ test-mir-pass: EarlyOtherwiseBranch
//@ compile-flags: -Zmir-enable-passes=+GVN,+SimplifyLocals-after-value-numbering
//@ needs-unwind

use std::task::Poll;

// We find a matching pattern in the unwind path,
// and we need to create a cleanup BB for this case to meet the unwind invariants rule.
// NB: This transform is not happening currently.

// EMIT_MIR early_otherwise_branch_unwind.unwind.EarlyOtherwiseBranch.diff
pub fn unwind<T>(val: Option<Option<Result<T, T>>>) {
    // CHECK-LABEL: fn unwind(
    // SHOULD-CHECK: drop({{.*}}) -> [return: bb{{.*}}, unwind: [[PARENT_UNWIND_BB:bb.*]]];
    // SHOULD-CHECK: [[PARENT_UNWIND_BB]] (cleanup): {
    // SHOULD-CHECK-NEXT: switchInt
    match val {
        Some(Some(Ok(_v))) => {}
        Some(Some(Err(_))) => {}
        Some(None) => {}
        None => {}
    }
}

// From https://github.com/rust-lang/rust/issues/130769#issuecomment-2370443086.
// EMIT_MIR early_otherwise_branch_unwind.poll.EarlyOtherwiseBranch.diff
pub fn poll(val: Poll<Result<Option<Vec<u8>>, u8>>) {
    // CHECK-LABEL: fn poll(
    // SHOULD-CHECK: drop({{.*}}) -> [return: bb{{.*}}, unwind: [[PARENT_UNWIND_BB:bb.*]]];
    // SHOULD-CHECK: [[PARENT_UNWIND_BB]] (cleanup): {
    // SHOULD-CHECK-NEXT: switchInt
    match val {
        Poll::Ready(Ok(Some(_trailers))) => {}
        Poll::Ready(Err(_err)) => {}
        Poll::Ready(Ok(None)) => {}
        Poll::Pending => {}
    }
}
