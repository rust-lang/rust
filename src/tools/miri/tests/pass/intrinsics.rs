//@compile-flags: -Zmiri-permissive-provenance
#![feature(core_intrinsics, layout_for_ptr)]
//! Tests for various intrinsics that do not fit anywhere else.

use std::intrinsics;
use std::mem::{size_of, size_of_val, size_of_val_raw, discriminant};

struct Bomb;

impl Drop for Bomb {
    fn drop(&mut self) {
        eprintln!("BOOM!");
    }
}

fn main() {
    assert_eq!(size_of::<Option<i32>>(), 8);
    assert_eq!(size_of_val(&()), 0);
    assert_eq!(size_of_val(&42), 4);
    assert_eq!(size_of_val(&[] as &[i32]), 0);
    assert_eq!(size_of_val(&[1, 2, 3] as &[i32]), 12);
    assert_eq!(size_of_val("foobar"), 6);

    unsafe {
        assert_eq!(size_of_val_raw(&[1] as &[i32] as *const [i32]), 4);
    }
    unsafe {
        assert_eq!(size_of_val_raw(0x100 as *const i32), 4);
    }

    assert_eq!(intrinsics::type_name::<Option<i32>>(), "core::option::Option<i32>");

    assert_eq!(intrinsics::likely(false), false);
    assert_eq!(intrinsics::unlikely(true), true);

    intrinsics::forget(Bomb);

    let _v = intrinsics::discriminant_value(&Some(()));
    let _v = intrinsics::discriminant_value(&0);
    let _v = intrinsics::discriminant_value(&true);
    let _v = intrinsics::discriminant_value(&vec![1, 2, 3]);
    // Make sure that even if the discriminant is stored together with data, the intrinsic returns
    // only the discriminant, nothing about the data.
    assert_eq!(discriminant(&Some(false)), discriminant(&Some(true)));
}
