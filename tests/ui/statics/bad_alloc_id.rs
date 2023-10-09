// aux-build: bad_alloc_id.rs
// run-pass

//! This test checks that we do not accidentally duplicate static items' base
//! allocation, even if we go through some projections. This is achieved by
//! never exposing a static item's "memory alloc id". Every static item has two
//! `AllocId`s: One which is backed by `GlobalAlloc::Static` and allows us to
//! figure out the static item. Then we can evaluate that static item, giving us
//! the static's memory-id, which is backed by `GlobalAlloc::Memory`. We always
//! immediately convert to the memory representation and throw away the memory
//! alloc id.

#![feature(const_mut_refs)]

extern crate bad_alloc_id;

static mut BAR: &mut u32 = unsafe {
    match &mut bad_alloc_id::FOO {
        Some(x) => x,
        None => panic!(),
    }
};

fn main() {
    unsafe {
        assert_eq!(BAR as *mut u32, bad_alloc_id::FOO.as_mut().unwrap() as *mut u32);
    }
}
