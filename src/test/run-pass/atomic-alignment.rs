// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(cfg_target_has_atomic)]
#![feature(integer_atomics)]

use std::mem::{align_of, size_of};
use std::sync::atomic::*;

fn main() {
    #[cfg(target_has_atomic = "8")]
    assert_eq!(align_of::<AtomicBool>(), size_of::<AtomicBool>());
    #[cfg(target_has_atomic = "ptr")]
    assert_eq!(align_of::<AtomicPtr<u8>>(), size_of::<AtomicPtr<u8>>());
    #[cfg(target_has_atomic = "8")]
    assert_eq!(align_of::<AtomicU8>(), size_of::<AtomicU8>());
    #[cfg(target_has_atomic = "8")]
    assert_eq!(align_of::<AtomicI8>(), size_of::<AtomicI8>());
    #[cfg(target_has_atomic = "16")]
    assert_eq!(align_of::<AtomicU16>(), size_of::<AtomicU16>());
    #[cfg(target_has_atomic = "16")]
    assert_eq!(align_of::<AtomicI16>(), size_of::<AtomicI16>());
    #[cfg(target_has_atomic = "32")]
    assert_eq!(align_of::<AtomicU32>(), size_of::<AtomicU32>());
    #[cfg(target_has_atomic = "32")]
    assert_eq!(align_of::<AtomicI32>(), size_of::<AtomicI32>());
    #[cfg(target_has_atomic = "64")]
    assert_eq!(align_of::<AtomicU64>(), size_of::<AtomicU64>());
    #[cfg(target_has_atomic = "64")]
    assert_eq!(align_of::<AtomicI64>(), size_of::<AtomicI64>());
    #[cfg(target_has_atomic = "128")]
    assert_eq!(align_of::<AtomicU128>(), size_of::<AtomicU128>());
    #[cfg(target_has_atomic = "128")]
    assert_eq!(align_of::<AtomicI128>(), size_of::<AtomicI128>());
    #[cfg(target_has_atomic = "ptr")]
    assert_eq!(align_of::<AtomicUsize>(), size_of::<AtomicUsize>());
    #[cfg(target_has_atomic = "ptr")]
    assert_eq!(align_of::<AtomicIsize>(), size_of::<AtomicIsize>());
}
