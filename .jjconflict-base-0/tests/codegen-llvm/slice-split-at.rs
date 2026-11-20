//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

// Check that no panic is generated in `split_at` when calculating the index for
// the tail chunk using `checked_sub`.
//
// Tests written for refactored implementations of:
// `<[T]>::{split_last_chunk, split_last_chunk_mut, last_chunk, last_chunk_mut}`

// CHECK-LABEL: @split_at_last_chunk
#[no_mangle]
pub fn split_at_last_chunk(s: &[u8], chunk_size: usize) -> Option<(&[u8], &[u8])> {
    // CHECK-NOT: panic
    let Some(index) = s.len().checked_sub(chunk_size) else { return None };
    Some(s.split_at(index))
}

// CHECK-LABEL: @split_at_mut_last_chunk
#[no_mangle]
pub fn split_at_mut_last_chunk(s: &mut [u8], chunk_size: usize) -> Option<(&mut [u8], &mut [u8])> {
    // CHECK-NOT: panic
    let Some(index) = s.len().checked_sub(chunk_size) else { return None };
    Some(s.split_at_mut(index))
}
