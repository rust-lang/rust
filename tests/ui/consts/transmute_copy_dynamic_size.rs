// Ensure const-eval correctly truncates an unsized source when the destination is smaller.

//@ run-pass
use std::mem::transmute_copy;

const BYTES: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const SLICE: &[u8] = &BYTES;
const SHRUNK: u32 = unsafe { transmute_copy::<[u8], u32>(SLICE) };

const _: () = assert!(SHRUNK == u32::from_ne_bytes([1, 2, 3, 4]));

fn main() {}
