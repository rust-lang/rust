//@run-rustfix

#![warn(clippy::deref_by_slicing)]
#![allow(clippy::borrow_deref_ref)]

use std::io::Read;

fn main() {
    let mut vec = vec![0];
    let _ = &vec[..];
    let _ = &mut vec[..];

    let ref_vec = &mut vec;
    let _ = &ref_vec[..];
    let mut_slice = &mut ref_vec[..];
    let _ = &mut mut_slice[..]; // Err, re-borrows slice

    let s = String::new();
    let _ = &s[..];

    static S: &[u8] = &[0, 1, 2];
    let _ = &mut &S[..]; // Err, re-borrows slice

    let slice: &[u32] = &[0u32, 1u32];
    let slice_ref = &slice;
    let _ = &slice_ref[..]; // Err, derefs slice

    let bytes: &[u8] = &[];
    let _ = (&bytes[..]).read_to_end(&mut vec![]).unwrap(); // Err, re-borrows slice
}
