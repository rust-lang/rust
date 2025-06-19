//@ run-pass
// Test that unsafe impl for Sync/Send can be provided for extern types.

#![feature(extern_types, sized_hierarchy)]

use std::marker::PointeeSized;

extern "C" {
    type A;
}

unsafe impl Sync for A {}
unsafe impl Send for A {}

fn assert_sync<T: PointeeSized + Sync>() {}
fn assert_send<T: PointeeSized + Send>() {}

fn main() {
    assert_sync::<A>();
    assert_send::<A>();
}
