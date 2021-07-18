// run-pass

#![allow(enum_intrinsics_non_enums)]

use std::mem;

enum ADT {
    First(u32, u32),
    Second(u64)
}

pub fn main() {
    assert!(mem::discriminant(&ADT::First(0,0)) == mem::discriminant(&ADT::First(1,1)));
    assert!(mem::discriminant(&ADT::Second(5))  == mem::discriminant(&ADT::Second(6)));
    assert!(mem::discriminant(&ADT::First(2,2)) != mem::discriminant(&ADT::Second(2)));

    let _ = mem::discriminant(&10);
    let _ = mem::discriminant(&"test");
}
