//@ test-mir-pass: EarlyOtherwiseBranch
//@ compile-flags: -Zmir-enable-passes=+GVN,+SimplifyLocals-after-value-numbering
//@ needs-unwind

use std::task::Poll;

// We find a matching pattern in the unwind path,
// and we need to create a cleanup BB for this case to meet the unwind invariants rule.
// NB: This transform is not happening currently.

// EMIT_MIR early_otherwise_branch_unwind.unwind.EarlyOtherwiseBranch.diff
fn unwind<T>(val: Option<Option<Option<T>>>) {
    // CHECK-LABEL: fn unwind(
    // CHECK: drop({{.*}}) -> [return: bb{{.*}}, unwind: [[PARENT_UNWIND_BB:bb.*]]];
    // CHECK: [[PARENT_UNWIND_BB]] (cleanup): {
    // After <https://github.com/rust-lang/rust/issues/155241>, GVN no longer
    // shares the discriminant locals across the move of `_v`, so
    // `EarlyOtherwiseBranch` introduces a fresh `discriminant` read here
    // before the `switchInt`.
    // CHECK-NEXT: discriminant
    // CHECK-NEXT: switchInt
    match val {
        Some(Some(Some(_v))) => {}
        Some(Some(None)) => {}
        Some(None) => {}
        None => {}
    }
}

// From https://github.com/rust-lang/rust/issues/130769#issuecomment-2370443086.
// EMIT_MIR early_otherwise_branch_unwind.poll.EarlyOtherwiseBranch.diff
pub fn poll(val: Poll<Result<Option<Vec<u8>>, u8>>) {
    // CHECK-LABEL: fn poll(
    // CHECK: drop({{.*}}) -> [return: bb{{.*}}, unwind: [[PARENT_UNWIND_BB:bb.*]]];
    // CHECK: [[PARENT_UNWIND_BB]] (cleanup): {
    // After <https://github.com/rust-lang/rust/issues/155241>, GVN no longer
    // shares the discriminant locals across the move of `_trailers`, so
    // `EarlyOtherwiseBranch` introduces a fresh `discriminant` read here
    // before the `switchInt`.
    // CHECK-NEXT: discriminant
    // CHECK-NEXT: switchInt
    match val {
        Poll::Ready(Ok(Some(_trailers))) => {}
        Poll::Ready(Err(_err)) => {}
        Poll::Ready(Ok(None)) => {}
        Poll::Pending => {}
    }
}
