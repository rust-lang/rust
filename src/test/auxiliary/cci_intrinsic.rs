// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod rusti {
    extern "rust-intrinsic" {
        pub fn atomic_cxchg(dst: &mut int, old: int, src: int) -> int;
        pub fn atomic_cxchg_acq(dst: &mut int, old: int, src: int) -> int;
        pub fn atomic_cxchg_rel(dst: &mut int, old: int, src: int) -> int;

        pub fn atomic_xchg(dst: &mut int, src: int) -> int;
        pub fn atomic_xchg_acq(dst: &mut int, src: int) -> int;
        pub fn atomic_xchg_rel(dst: &mut int, src: int) -> int;

        pub fn atomic_xadd(dst: &mut int, src: int) -> int;
        pub fn atomic_xadd_acq(dst: &mut int, src: int) -> int;
        pub fn atomic_xadd_rel(dst: &mut int, src: int) -> int;

        pub fn atomic_xsub(dst: &mut int, src: int) -> int;
        pub fn atomic_xsub_acq(dst: &mut int, src: int) -> int;
        pub fn atomic_xsub_rel(dst: &mut int, src: int) -> int;
    }
}

#[inline(always)]
pub fn atomic_xchg(dst: &mut int, src: int) -> int {
    unsafe {
        rusti::atomic_xchg(dst, src)
    }
}
