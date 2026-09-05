// Sanity checks for `transmute_copy` across different source/destination layouts,
// including scalars, scalar pairs, size differences, large values, and wide pointers.


//@ run-pass

use std::mem::transmute_copy;
use std::convert::TryInto;

// Same size scalar transmute.
fn scalar_exact_size() {
    let x: i32 = -42;
    let y: u32 = unsafe { transmute_copy(&x) };
    assert_eq!(y, x as u32);
}

// Same size pair of scalars transmute.
fn scalar_pair_exact_size() {
    let x: (i64, i64) = (1, -2);
    let y: (u64, u64) = unsafe { transmute_copy(&x) };
    assert_eq!(y, (1u64, (-2i64) as u64));
}

// Copying from a larger source into a smaller destination.
fn shrinking_copy() {
    let pair: (u64, u64) = (0x1122_3344_5566_7788, 0x99aa_bbcc_ddee_ff00);
    let x: u64 = unsafe { transmute_copy(&pair) };
    assert_eq!(x, pair.0);
}

// Large memory representation value, exercising the memcpy path.
fn large_array() {
    let src = [0x42u8; 4096];
    let x: [u8; 4096] = unsafe { transmute_copy(&src) };
    assert_eq!(x, src);
}

// Unsized slice source: ensure the data pointer, rather than metadata, is used.
fn slice_source() {
    let bytes: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8];
    let x: u64 = unsafe { transmute_copy(bytes) };
    assert_eq!(x, u64::from_ne_bytes(bytes.try_into().unwrap()));
}

// Unsized `str` source: ensure the data pointer is used rather than metadata.
fn str_source() {
    let s: &str = "abcdefgh";
    let x: u64 = unsafe { transmute_copy(s) };
    assert_eq!(x, u64::from_ne_bytes(s.as_bytes().try_into().unwrap()));
}

// Trait object source: ensure the data pointer is used rather than the vtable pointer.
trait Foo {}
impl Foo for u64 {}
fn dyn_trait_source() {
    let v: u64 = 0x0102030405060708;
    let obj: &dyn Foo = &v;
    let x: u64 = unsafe { transmute_copy(obj) };
    assert_eq!(x, v);
}

fn main() {
    scalar_exact_size();
    scalar_pair_exact_size();
    shrinking_copy();
    large_array();
    slice_source();
    str_source();
    dyn_trait_source();
}
