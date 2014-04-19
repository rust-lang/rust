// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Issue #2303

use std::mem;

mod rusti {
    extern "rust-intrinsic" {
        pub fn pref_align_of<T>() -> uint;
        pub fn min_align_of<T>() -> uint;
    }
}

// This is the type with the questionable alignment
struct Inner {
    c64: u64
}

// This is the type that contains the type with the
// questionable alignment, for testing
struct Outer {
    c8: u8,
    t: Inner
}


#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
mod m {
    #[cfg(target_arch = "x86")]
    pub mod m {
        pub fn align() -> uint { 4u }
        pub fn size() -> uint { 12u }
    }

    #[cfg(target_arch = "x86_64")]
    pub mod m {
        pub fn align() -> uint { 8u }
        pub fn size() -> uint { 16u }
    }
}

#[cfg(target_os = "win32")]
mod m {
    #[cfg(target_arch = "x86")]
    pub mod m {
        pub fn align() -> uint { 8u }
        pub fn size() -> uint { 16u }
    }
}

#[cfg(target_os = "android")]
mod m {
    #[cfg(target_arch = "arm")]
    pub mod m {
        pub fn align() -> uint { 8u }
        pub fn size() -> uint { 16u }
    }
}

pub fn main() {
    unsafe {
        let x = Outer {c8: 22u8, t: Inner {c64: 44u64}};

        // Send it through the shape code
        let y = format!("{:?}", x);

        println!("align inner = {}", rusti::min_align_of::<Inner>());
        println!("size outer = {}", mem::size_of::<Outer>());
        println!("y = {}", y);

        // per clang/gcc the alignment of `Inner` is 4 on x86.
        assert_eq!(rusti::min_align_of::<Inner>(), m::m::align());

        // per clang/gcc the size of `Outer` should be 12
        // because `Inner`s alignment was 4.
        assert_eq!(mem::size_of::<Outer>(), m::m::size());

        assert_eq!(y, "Outer{c8: 22u8, t: Inner{c64: 44u64}}".to_owned());
    }
}
