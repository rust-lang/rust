// unit-test: NormalizeArrayLen
// compile-flags: -Zmir-enable-passes=+LowerSliceLenCalls

// EMIT_MIR lower_array_len.array_bound.NormalizeArrayLen.diff
pub fn array_bound<const N: usize>(index: usize, slice: &[u8; N]) -> u8 {
    if index < slice.len() {
        slice[index]
    } else {
        42
    }
}

// EMIT_MIR lower_array_len.array_bound_mut.NormalizeArrayLen.diff
pub fn array_bound_mut<const N: usize>(index: usize, slice: &mut [u8; N]) -> u8 {
    if index < slice.len() {
        slice[index]
    } else {
        slice[0] = 42;

        42
    }
}

// EMIT_MIR lower_array_len.array_len.NormalizeArrayLen.diff
pub fn array_len<const N: usize>(arr: &[u8; N]) -> usize {
    arr.len()
}

// EMIT_MIR lower_array_len.array_len_by_value.NormalizeArrayLen.diff
pub fn array_len_by_value<const N: usize>(arr: [u8; N]) -> usize {
    arr.len()
}

fn main() {
    let _ = array_bound(3, &[0, 1, 2, 3]);
    let mut tmp = [0, 1, 2, 3, 4];
    let _ = array_bound_mut(3, &mut [0, 1, 2, 3]);
    let _ = array_len(&[0]);
    let _ = array_len_by_value([0, 2]);
}
