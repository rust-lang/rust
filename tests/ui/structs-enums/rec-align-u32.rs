//@ run-pass
#![allow(dead_code)]
#![allow(unused_unsafe)]
// Issue #2303

#![feature(core_intrinsics, rustc_attrs)]

use std::mem;
use std::intrinsics;

// This is the type with the questionable alignment
#[derive(Debug)]
struct Inner {
    c64: u32
}

// This is the type that contains the type with the
// questionable alignment, for testing
#[derive(Debug)]
struct Outer {
    c8: u8,
    t: Inner
}

mod m {
    pub fn align() -> usize { 4 }
    pub fn size() -> usize { 8 }
}

pub fn main() {
    unsafe {
        let x = Outer {c8: 22, t: Inner {c64: 44}};

        // Send it through the shape code
        let y = format!("{:?}", x);

        println!("align inner = {:?}", intrinsics::min_align_of::<Inner>());
        println!("size outer = {:?}", mem::size_of::<Outer>());
        println!("y = {:?}", y);

        // per clang/gcc the alignment of `inner` is 4 on x86.
        assert_eq!(intrinsics::min_align_of::<Inner>(), m::align());

        // per clang/gcc the size of `outer` should be 12
        // because `inner`s alignment was 4.
        assert_eq!(mem::size_of::<Outer>(), m::size());

        assert_eq!(y, "Outer { c8: 22, t: Inner { c64: 44 } }".to_string());
    }
}
