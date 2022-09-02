// compile-flags: -Z mir-opt-level=4

// EMIT_MIR lower_array_len_e2e.array_bound.PreCodegen.after.mir
pub fn array_bound<const N: usize>(index: usize, slice: &[u8; N]) -> u8 {
    if index < slice.len() {
        slice[index]
    } else {
        42
    }
}

// EMIT_MIR lower_array_len_e2e.array_bound_mut.PreCodegen.after.mir
pub fn array_bound_mut<const N: usize>(index: usize, slice: &mut [u8; N]) -> u8 {
    if index < slice.len() {
        slice[index]
    } else {
        slice[0] = 42;

        42
    }
}

// EMIT_MIR lower_array_len_e2e.array_len.PreCodegen.after.mir
pub fn array_len<const N: usize>(arr: &[u8; N]) -> usize {
    arr.len()
}

// EMIT_MIR lower_array_len_e2e.array_len_by_value.PreCodegen.after.mir
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
