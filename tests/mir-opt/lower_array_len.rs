// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: GVN
//@ compile-flags: -Zmir-enable-passes=+LowerSliceLenCalls

// EMIT_MIR lower_array_len.array_bound.GVN.diff
pub fn array_bound<const N: usize>(index: usize, slice: &[u8; N]) -> u8 {
    // CHECK-LABEL: fn array_bound(
    // CHECK-NOT: Lt
    // CHECK: Lt(copy _1, const N);
    // CHECK-NOT: Lt
    if index < slice.len() { slice[index] } else { 42 }
}

// EMIT_MIR lower_array_len.array_bound_mut.GVN.diff
pub fn array_bound_mut<const N: usize>(index: usize, slice: &mut [u8; N]) -> u8 {
    // CHECK-LABEL: fn array_bound_mut(
    // CHECK-NOT: Lt
    // CHECK: Lt(copy _1, const N);
    // CHECK-NOT: Lt
    // CHECK: Lt(const 0_usize, const N)
    // CHECK-NOT: Lt
    if index < slice.len() {
        slice[index]
    } else {
        slice[0] = 42;

        42
    }
}

// EMIT_MIR lower_array_len.array_len.GVN.diff
pub fn array_len<const N: usize>(arr: &[u8; N]) -> usize {
    // CHECK-LABEL: fn array_len(
    // CHECK: _0 = const N;
    arr.len()
}

// EMIT_MIR lower_array_len.array_len_by_value.GVN.diff
pub fn array_len_by_value<const N: usize>(arr: [u8; N]) -> usize {
    // CHECK-LABEL: fn array_len_by_value(
    // CHECK: _0 = const N;
    arr.len()
}

// EMIT_MIR lower_array_len.array_len_reborrow.GVN.diff
pub fn array_len_reborrow<const N: usize>(mut arr: [u8; N]) -> usize {
    // CHECK-LABEL: fn array_len_reborrow(
    // CHECK: _0 = const N;
    let arr: &mut [_] = &mut arr;
    let arr = &*arr;
    arr.len()
}

// EMIT_MIR lower_array_len.array_len_raw.GVN.diff
pub fn array_len_raw<const N: usize>(arr: [u8; N]) -> usize {
    // CHECK-LABEL: fn array_len_raw(
    // CHECK: _0 = const N;
    let arr: &[_] = &arr;
    let arr = std::ptr::addr_of!(*arr);
    unsafe { &*arr }.len()
}

fn main() {
    let _ = array_bound(3, &[0, 1, 2, 3]);
    let mut tmp = [0, 1, 2, 3, 4];
    let _ = array_bound_mut(3, &mut [0, 1, 2, 3]);
    let _ = array_len(&[0]);
    let _ = array_len_by_value([0, 2]);
    let _ = array_len_reborrow([0, 2]);
    let _ = array_len_raw([0, 2]);
}
