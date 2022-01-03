// run-rustfix

#![allow(unused)]
#![warn(clippy::redundant_slicing)]

fn main() {
    let slice: &[u32] = &[0];
    let _ = &slice[..]; // Redundant slice

    let v = vec![0];
    let _ = &v[..]; // Deref instead of slice
    let _ = &(&*v)[..]; // Outer borrow is redundant

    static S: &[u8] = &[0, 1, 2];
    let err = &mut &S[..]; // Should reborrow instead of slice

    let mut vec = vec![0];
    let mut_slice = &mut vec[..]; // Deref instead of slice
    let _ = &mut mut_slice[..]; // Should reborrow instead of slice

    let ref_vec = &vec;
    let _ = &ref_vec[..]; // Deref instead of slice

    macro_rules! m {
        ($e:expr) => {
            $e
        };
    }
    let _ = &m!(slice)[..];

    macro_rules! m2 {
        ($e:expr) => {
            &$e[..]
        };
    }
    let _ = m2!(slice); // Don't lint in a macro

    let slice_ref = &slice;
    let _ = &slice_ref[..]; // Deref instead of slice
}
