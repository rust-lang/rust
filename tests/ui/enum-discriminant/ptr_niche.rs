//@ run-pass
//! Check that we can codegen setting and getting discriminants, including non-null niches,
//! for enums with a pointer-like ABI. This used to crash llvm.

#![feature(rustc_attrs)]
use std::{ptr, mem};


#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_layout_scalar_valid_range_end(100)]
#[derive(Copy, Clone)]
struct PointerWithRange(#[allow(dead_code)] *const u8);


fn main() {
    let val = unsafe { PointerWithRange(ptr::without_provenance(90)) };

    let ptr = Some(val);
    assert!(ptr.is_some());
    let raw = unsafe { mem::transmute::<_, usize>(ptr) };
    assert_eq!(raw, 90);

    let ptr = Some(Some(val));
    assert!(ptr.is_some());
    assert!(ptr.unwrap().is_some());
    let raw = unsafe { mem::transmute::<_, usize>(ptr) };
    assert_eq!(raw, 90);

    let ptr: Option<PointerWithRange> = None;
    assert!(ptr.is_none());
    let raw = unsafe { mem::transmute::<_, usize>(ptr) };
    assert!(!(1..=100).contains(&raw));

    let ptr: Option<Option<PointerWithRange>> = None;
    assert!(ptr.is_none());
    let raw = unsafe { mem::transmute::<_, usize>(ptr) };
    assert!(!(1..=100).contains(&raw));
}
