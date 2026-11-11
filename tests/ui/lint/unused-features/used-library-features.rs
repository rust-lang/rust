//@ check-pass

#![crate_type = "lib"]
#![deny(unused_features)]

// Used library features
#![feature(error_iter)]
#![cfg_attr(all(), feature(allocator_ext))]

pub fn use_error_iter(e: &(dyn std::error::Error + 'static)) {
    for _ in e.sources() {}
}

pub fn use_allocator_api() {
    use std::alloc::Global;
    let _ = Vec::<i32>::try_with_capacity_in(1, Global);
}
