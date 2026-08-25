//@ test-mir-pass: LowerSliceLenCalls
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// EMIT_MIR lower_slice_len.bound.LowerSliceLenCalls.diff
pub fn bound(index: usize, slice: &[u8]) -> u8 {
    // CHECK-LABEL: fn bound(
    // CHECK-NOT: ::len(
    if index < slice.len() { slice[index] } else { 42 }
}

fn main() {
    let _ = bound(1, &[1, 2, 3]);
}
