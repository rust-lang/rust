//@ run-pass

use std::mem::transmute_copy;

// Test exact size `transmute_copy` from dynamically sized types. Both sources
// have a runtime size of 8 bytes, exactly matching the `u64` destination.
const BYTES: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const SLICE: &[u8] = &BYTES;
const EXACT: u64 = unsafe { transmute_copy::<[u8], u64>(SLICE) }; // dynamic len == dst size

// `str` is distinct from `[T]` and carries its size in its own metadata. Keep
// it as a separate case to ensure the exact size handling is not specific to
// slice types.
const S: &str = "abcdefgh";
const EXACT_STR: u64 = unsafe { transmute_copy::<str, u64>(S) };

const _: () = assert!(EXACT == u64::from_ne_bytes(BYTES));
const _: () = assert!(EXACT_STR == u64::from_ne_bytes(*S.as_bytes().first_chunk().unwrap()));

fn main() {}
