// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: LowerSliceLenCalls

// EMIT_MIR lower_slice_len.bound.LowerSliceLenCalls.diff
pub fn bound(index: usize, slice: &[u8]) -> u8 {
    if index < slice.len() {
        slice[index]
    } else {
        42
    }
}

fn main() {
    let _ = bound(1, &[1, 2, 3]);
}
