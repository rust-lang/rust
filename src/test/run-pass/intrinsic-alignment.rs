// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(intrinsics)]

mod rusti {
    extern "rust-intrinsic" {
        pub fn pref_align_of<T>() -> uint;
        pub fn min_align_of<T>() -> uint;
    }
}

#[cfg(any(target_os = "linux",
          target_os = "macos",
          target_os = "freebsd",
          target_os = "dragonfly"))]
mod m {
    #[main]
    #[cfg(target_arch = "x86")]
    pub fn main() {
        unsafe {
            assert_eq!(::rusti::pref_align_of::<u64>(), 8u);
            assert_eq!(::rusti::min_align_of::<u64>(), 4u);
        }
    }

    #[main]
    #[cfg(any(target_arch = "x86_64", target_arch = "arm"))]
    pub fn main() {
        unsafe {
            assert_eq!(::rusti::pref_align_of::<u64>(), 8u);
            assert_eq!(::rusti::min_align_of::<u64>(), 8u);
        }
    }
}

#[cfg(target_os = "windows")]
mod m {
    #[main]
    #[cfg(target_arch = "x86")]
    pub fn main() {
        unsafe {
            assert_eq!(::rusti::pref_align_of::<u64>(), 8u);
            assert_eq!(::rusti::min_align_of::<u64>(), 8u);
        }
    }

    #[main]
    #[cfg(target_arch = "x86_64")]
    pub fn main() {
        unsafe {
            assert_eq!(::rusti::pref_align_of::<u64>(), 8u);
            assert_eq!(::rusti::min_align_of::<u64>(), 8u);
        }
    }
}

#[cfg(target_os = "android")]
mod m {
    #[main]
    #[cfg(target_arch = "arm")]
    pub fn main() {
        unsafe {
            assert_eq!(::rusti::pref_align_of::<u64>(), 8u);
            assert_eq!(::rusti::min_align_of::<u64>(), 8u);
        }
    }
}
