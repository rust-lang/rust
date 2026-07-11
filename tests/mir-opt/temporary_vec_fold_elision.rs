//@ compile-flags: -Zmir-opt-level=2 -Cpanic=abort
//@ skip-filecheck

// EMIT_MIR temporary_vec_fold_elision.wrapping_sum.TemporaryVecFoldElision.diff
pub fn wrapping_sum(nums: &[u64]) -> u64 {
    let values: Vec<u64> = nums
        .iter()
        .filter(|n| **n != 0)
        .copied()
        .enumerate()
        .map(|(index, n)| n.wrapping_add(index as u64))
        .collect();
    values.into_iter().fold(0, u64::wrapping_add)
}

// EMIT_MIR temporary_vec_fold_elision.capturing_map.TemporaryVecFoldElision.after.mir
pub fn capturing_map(nums: &[u64], offset: u64) -> u64 {
    let values: Vec<u64> = nums.iter().map(|n| n.wrapping_add(offset)).collect();
    values.into_iter().fold(0, u64::wrapping_add)
}
