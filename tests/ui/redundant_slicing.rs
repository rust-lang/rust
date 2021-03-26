#![allow(unused)]
#![warn(clippy::redundant_slicing)]

fn main() {
    let slice: &[u32] = &[0];
    let _ = &slice[..];

    let v = vec![0];
    let _ = &v[..]; // Changes the type
    let _ = &(&v[..])[..]; // Outer borrow is redundant

    static S: &[u8] = &[0, 1, 2];
    let err = &mut &S[..]; // Should reborrow instead of slice

    let mut vec = vec![0];
    let mut_slice = &mut *vec;
    let _ = &mut mut_slice[..]; // Should reborrow instead of slice

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
}
