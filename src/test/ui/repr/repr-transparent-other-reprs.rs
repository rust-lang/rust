// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(repr_align, attr_literals)]

// See also repr-transparent.rs

#[repr(transparent, C)] //~ ERROR cannot have other repr
struct TransparentPlusC {
    ptr: *const u8
}

#[repr(transparent, packed)] //~ ERROR cannot have other repr
struct TransparentPlusPacked(*const u8);

#[repr(transparent, align(2))] //~ ERROR cannot have other repr
struct TransparentPlusAlign(u8);

#[repr(transparent)] //~ ERROR cannot have other repr
#[repr(C)]
struct SeparateAttributes(*mut u8);
