// compile-flags: -Z mir-opt-level=4 -Zunsound-mir-opts

// must not optimize as it does not follow the pattern of
// left and right hand side being the same variant

// EMIT_MIR early_otherwise_branch_noopt.noopt1.EarlyOtherwiseBranch.diff
fn noopt1(x: Option<u32>, y: Option<u32>) -> u32 {
    match (x, y) {
        (Some(a), Some(b)) => 0,
        (Some(a), None) => 1,
        (None, Some(b)) => 2,
        (None, None) => 3,
    }
}

// must not optimize as the types being matched on are not identical
// EMIT_MIR early_otherwise_branch_noopt.noopt2.EarlyOtherwiseBranch.diff
fn noopt2(x: Option<u32>, y: Option<bool>) -> u32 {
    match (x, y) {
        (Some(a), Some(b)) => 0,
        _ => 1,
    }
}

fn main() {
    noopt1(None, Some(0));
    noopt2(None, Some(true));
}
