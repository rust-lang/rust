//@ build-fail
//@ revisions: default optimized
//@ compile-flags: -Znext-solver
//@[optimized] compile-flags: -O

#![feature(gca_min_const_items, generic_const_items, gca_const_items)]
#![expect(incomplete_features)]

//~? ERROR too big for the target architecture

const N<T>: usize = 1 + size_of::<T>();

struct Buffer<const CAP: usize>([u8; CAP]);

impl<const CAP: usize> Buffer<CAP> {
    fn new() -> Self {
        Self([0; CAP])
    }
}

fn trigger<T>() {
    let _ = Buffer::<{ core::gca!(N::<T>) }>::new();
}

fn main() {
    trigger::<[u8; usize::MAX]>();
}
