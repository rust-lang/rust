// run-pass
#![allow(dead_code)]
#![allow(deprecated)]

use std::mem;

enum Tag<A> {
    Tag2(A)
}

struct Rec {
    c8: u8,
    t: Tag<u64>
}

fn mk_rec() -> Rec {
    return Rec { c8:0, t:Tag::Tag2(0) };
}

fn is_u64_aligned(u: &Tag<u64>) -> bool {
    let p: usize = unsafe { mem::transmute(u) };
    let u64_align = std::mem::min_align_of::<u64>();
    return (p & (u64_align - 1)) == 0;
}

pub fn main() {
    let x = mk_rec();
    assert!(is_u64_aligned(&x.t));
}
