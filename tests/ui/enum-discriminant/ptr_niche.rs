//@ run-pass
//! Check that we can codegen setting and getting discriminants, including non-null niches,
//! for enums with a pointer-like ABI. This used to crash llvm.

#![feature(rustc_attrs, pattern_types, pattern_type_macro)]
use std::{ptr, mem, pat::pattern_type};

#[derive(Copy, Clone)]
struct PointerWithRange(#[allow(dead_code)] pattern_type!(*const u8 is !null));


fn main() {
    let val =
        unsafe { PointerWithRange(mem::transmute::<*const u8, _>(ptr::without_provenance(90))) };

    let ptr = Some(val);
    assert!(ptr.is_some());
    let raw = unsafe { mem::transmute::<_, usize>(ptr) };
    assert_eq!(raw, 90);

    /*
    FIXME(pattern_types): allow restricting raw pointers to smaller integer ranges?
    let ptr = Some(Some(val));
    assert!(ptr.is_some());
    assert!(ptr.unwrap().is_some());
    let raw = unsafe { mem::transmute::<_, usize>(ptr) };
    assert_eq!(raw, 90);
    */

    let ptr: Option<PointerWithRange> = None;
    assert!(ptr.is_none());
    let raw = unsafe { mem::transmute::<_, usize>(ptr) };
    assert!(!(1..=100).contains(&raw));

    /*
    FIXME(pattern_types): allow restricting raw pointers to smaller integer ranges?
    let ptr: Option<Option<PointerWithRange>> = None;
    assert!(ptr.is_none());
    let raw = unsafe { mem::transmute::<_, usize>(ptr) };
    assert!(!(1..=100).contains(&raw));
    */
}
