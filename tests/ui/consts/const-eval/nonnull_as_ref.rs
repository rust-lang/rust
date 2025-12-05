//@ check-pass

use std::ptr::NonNull;

const NON_NULL: NonNull<u8> = unsafe { NonNull::new_unchecked((&42u8 as *const u8).cast_mut()) };
const _: () = assert!(42 == *unsafe { NON_NULL.as_ref() });

fn main() {}
