// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: The Rust Project Developers (see https://thanks.rust-lang.org)

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
