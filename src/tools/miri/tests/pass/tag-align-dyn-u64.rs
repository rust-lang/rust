// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

enum Tag<A> {
    Tag2(A),
}

#[allow(dead_code)]
struct Rec {
    c8: u8,
    t: Tag<u64>,
}

fn mk_rec() -> Rec {
    return Rec { c8: 0, t: Tag::Tag2(0) };
}

fn is_u64_aligned(u: &Tag<u64>) -> bool {
    let p: *const () = unsafe { mem::transmute(u) };
    let p = p as usize;
    let u64_align = std::mem::align_of::<u64>();
    return (p & (u64_align - 1)) == 0;
}

pub fn main() {
    let x = mk_rec();
    assert!(is_u64_aligned(&x.t));
}
