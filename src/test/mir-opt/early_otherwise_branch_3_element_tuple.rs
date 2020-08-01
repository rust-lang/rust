// compile-flags: -Z mir-opt-level=3

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR early_otherwise_branch_3_element_tuple.opt1.EarlyOtherwiseBranch.diff
fn opt1(x: Option<usize>, y:Option<usize>, z:Option<usize>) -> usize {
    match (x,y,z) {
        (Some(a), Some(b), Some(c)) => 0,
        _ => 1
    }
}

fn main() {
    opt1(None, Some(0), None);
}
