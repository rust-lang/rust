//@ test-mir-pass: EarlyOtherwiseBranch
//@ compile-flags: -Zmir-enable-passes=+UnreachableEnumBranching

// Tests various cases that the `early_otherwise_branch` opt should *not* optimize

// From #78496
enum E<'a> {
    Empty,
    Some(&'a E<'a>),
}

// EMIT_MIR early_otherwise_branch_soundness.no_downcast.EarlyOtherwiseBranch.diff
fn no_downcast(e: &E) -> u32 {
    // CHECK-LABEL: fn no_downcast(
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK-NOT: Ne
    // CHECK-NOT: discriminant
    // CHECK: switchInt(move [[LOCAL1]]) -> [
    // CHECK-NEXT: }
    if let E::Some(E::Some(_)) = e { 1 } else { 2 }
}

// SAFETY: if `a` is `Some`, `b` must point to a valid, initialized value
// EMIT_MIR early_otherwise_branch_soundness.no_deref_ptr.EarlyOtherwiseBranch.diff
unsafe fn no_deref_ptr(a: Option<i32>, b: *const Option<i32>) -> i32 {
    // CHECK-LABEL: fn no_deref_ptr(
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK-NOT: Ne
    // CHECK-NOT: discriminant
    // CHECK: switchInt(move [[LOCAL1]]) -> [
    // CHECK-NEXT: }
    match a {
        // `*b` being correct depends on `a == Some(_)`
        Some(_) => match *b {
            Some(v) => v,
            _ => 0,
        },
        _ => 0,
    }
}

fn main() {
    no_downcast(&E::Empty);
    unsafe { no_deref_ptr(None, std::ptr::null()) };
}
