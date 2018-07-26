// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[repr(C)]
#[derive(Copy, Clone)]
struct SliceRepr {
    ptr: *const u8,
    len: usize,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct BadSliceRepr {
    ptr: *const u8,
    len: &'static u8,
}

union SliceTransmute {
    repr: SliceRepr,
    bad: BadSliceRepr,
    slice: &'static [u8],
    str: &'static str,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct DynRepr {
    ptr: *const u8,
    vtable: *const u8,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct DynRepr2 {
    ptr: *const u8,
    vtable: *const u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct BadDynRepr {
    ptr: *const u8,
    vtable: usize,
}

union DynTransmute {
    repr: DynRepr,
    repr2: DynRepr2,
    bad: BadDynRepr,
    rust: &'static Trait,
}

trait Trait {}

// OK
const A: &str = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 1 } }.str};
// should lint
const B: &str = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 999 } }.str};
// bad
const C: &str = unsafe { SliceTransmute { bad: BadSliceRepr { ptr: &42, len: &3 } }.str};
//~^ ERROR this constant likely exhibits undefined behavior

// OK
const A2: &[u8] = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 1 } }.slice};
// should lint
const B2: &[u8] = unsafe { SliceTransmute { repr: SliceRepr { ptr: &42, len: 999 } }.slice};
// bad
const C2: &[u8] = unsafe { SliceTransmute { bad: BadSliceRepr { ptr: &42, len: &3 } }.slice};
//~^ ERROR this constant likely exhibits undefined behavior

// bad
const D: &Trait = unsafe { DynTransmute { repr: DynRepr { ptr: &92, vtable: &3 } }.rust};
//~^ ERROR this constant likely exhibits undefined behavior
// bad
const E: &Trait = unsafe { DynTransmute { repr2: DynRepr2 { ptr: &92, vtable: &3 } }.rust};
//~^ ERROR this constant likely exhibits undefined behavior
// bad
const F: &Trait = unsafe { DynTransmute { bad: BadDynRepr { ptr: &92, vtable: 3 } }.rust};
//~^ ERROR this constant likely exhibits undefined behavior

fn main() {
}
