// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #2303

mod rusti {
    #[abi = "rust-intrinsic"]
    pub extern {
        pub fn pref_align_of<T>() -> uint;
        pub fn min_align_of<T>() -> uint;
    }
}

// This is the type with the questionable alignment
struct Inner {
    c64: u32
}

// This is the type that contains the type with the
// questionable alignment, for testing
struct Outer {
    c8: u8,
    t: Inner
}


#[cfg(target_arch = "x86")]
mod m {
    pub fn align() -> uint { 4u }
    pub fn size() -> uint { 8u }
}

#[cfg(target_arch = "x86_64")]
mod m {
    pub fn align() -> uint { 4u }
    pub fn size() -> uint { 8u }
}

pub fn main() {
    unsafe {
        let x = Outer {c8: 22u8, t: Inner {c64: 44u32}};

        // Send it through the shape code
        let y = fmt!("%?", x);

        debug!("align inner = %?", rusti::min_align_of::<Inner>());
        debug!("size outer = %?", sys::size_of::<Outer>());
        debug!("y = %s", y);

        // per clang/gcc the alignment of `inner` is 4 on x86.
        assert!(rusti::min_align_of::<Inner>() == m::align());

        // per clang/gcc the size of `outer` should be 12
        // because `inner`s alignment was 4.
        assert!(sys::size_of::<Outer>() == m::size());

        assert!(y == ~"{c8: 22, t: {c64: 44}}");
    }
}
