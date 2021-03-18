// compile-flags: -Z mir-opt-level=4

// EMIT_MIR early_otherwise_branch_3_element_tuple.opt1.EarlyOtherwiseBranch.diff
fn opt1(x: Option<u32>, y: Option<u32>, z: Option<u32>) -> u32 {
    match (x, y, z) {
        (Some(a), Some(b), Some(c)) => 0,
        _ => 1,
    }
}

fn main() {
    opt1(None, Some(0), None);
}
