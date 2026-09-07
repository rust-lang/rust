//@ compile-flags: -Zmir-opt-level=3 -Zmir-enable-passes=+CaptureMutVec
//@ test-mir-pass: CaptureMutVec
//@ skip-filecheck

// EMIT_MIR capture_mut_vec.push.CaptureMutVec.diff
#[inline(never)]
pub fn push(vec: &mut Vec<usize>, count: usize) {
    for value in 0..count {
        vec.push(value);
    }
}

fn main() {}
