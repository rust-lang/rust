//@ check-pass

#![feature(sync_unsafe_cell)]
#![allow(unused)]
#![deny(repr_transparent_external_private_fields)]

// https://github.com/rust-lang/rust/issues/129470

struct ZST;

#[repr(transparent)]
struct TransparentWithManuallyDropZST {
    value: i32,
    md: std::mem::ManuallyDrop<ZST>,
    mu: std::mem::MaybeUninit<ZST>,
    p: std::pin::Pin<ZST>,
    pd: std::marker::PhantomData<ZST>,
    pp: std::marker::PhantomPinned,
    c: std::cell::Cell<ZST>,
    uc: std::cell::UnsafeCell<ZST>,
    suc: std::cell::SyncUnsafeCell<ZST>,
    zst: ZST,
}

fn main() {}
