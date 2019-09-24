// build-pass (FIXME(62277): could be check-pass?)
#![feature(core_intrinsics)]
#![allow(warnings)]

use std::intrinsics;

#[derive(Copy, Clone)]
struct Wrap(i64);

// These volatile intrinsics used to cause an ICE

unsafe fn test_bool(p: &mut bool, v: bool) {
    intrinsics::volatile_load(p);
    intrinsics::volatile_store(p, v);
}

unsafe fn test_immediate_fca(p: &mut Wrap, v: Wrap) {
    intrinsics::volatile_load(p);
    intrinsics::volatile_store(p, v);
}

fn main() {}
