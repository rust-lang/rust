#![crate_type = "lib"]

// EMIT_MIR simplify_debuginfo_moves.test_other.SimplifyDebugInfo.diff
pub fn test_other(x: &Vec<usize>) -> usize {
    x.iter().collect::<Vec<_>>().len()
}
