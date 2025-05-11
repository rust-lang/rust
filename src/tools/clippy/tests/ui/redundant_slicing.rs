#![allow(unused, clippy::deref_by_slicing)]
#![warn(clippy::redundant_slicing)]

use std::io::Read;

fn main() {
    let slice: &[u32] = &[0];
    let _ = &slice[..]; // Redundant slice
    //
    //~^^ redundant_slicing

    let v = vec![0];
    let _ = &v[..]; // Ok, results in `&[_]`
    let _ = &(&*v)[..]; // Outer borrow is redundant
    //
    //~^^ redundant_slicing

    static S: &[u8] = &[0, 1, 2];
    let _ = &mut &S[..]; // Ok, re-borrows slice

    let mut vec = vec![0];
    let mut_slice = &mut vec[..]; // Ok, results in `&mut [_]`
    let _ = &mut mut_slice[..]; // Ok, re-borrows slice

    let ref_vec = &vec;
    let _ = &ref_vec[..]; // Ok, results in `&[_]`

    macro_rules! m {
        ($e:expr) => {
            $e
        };
    }
    let _ = &m!(slice)[..];
    //~^ redundant_slicing

    macro_rules! m2 {
        ($e:expr) => {
            &$e[..]
        };
    }
    let _ = m2!(slice); // Don't lint in a macro

    let slice_ref = &slice;
    let _ = &slice_ref[..]; // Ok, derefs slice

    // Issue #7972
    let bytes: &[u8] = &[];
    let _ = (&bytes[..]).read_to_end(&mut vec![]).unwrap(); // Ok, re-borrows slice
}
