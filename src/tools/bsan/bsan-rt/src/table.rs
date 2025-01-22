#![warn(clippy::pedantic)]

use macros::init_statics;
use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::ops::BitAnd;

init_statics!(X86, 48);

pub trait Provenance: Copy + Sized {}

#[repr(C)]
struct L2<const W: usize, T: Provenance> {
    bytes: *mut u8,
    phantom: PhantomData<T>,
}

impl<const W: usize, T: Provenance> L2<W, T> {
    fn new() -> Self {
        unsafe {
            let bytes = alloc(Layout::from_size_align(W, 1).unwrap());
            Self {
                bytes,
                phantom: PhantomData,
            }
        }
    }
    #[inline]
    fn lookup(&mut self, index: usize) -> &mut T {
        debug_assert!(index < W);
        unsafe {
            let bytes = self.bytes.add(index);
            let bytes = bytes.cast::<T>();
            &mut *bytes
        }
    }
}

impl<const W: usize, T: Provenance> Drop for L2<W, T> {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align(W, 1).unwrap();
            dealloc(self.bytes, layout);
        }
    }
}

#[repr(C)]
struct L1<const W1: usize, const W2: usize, T: Provenance> {
    entries: *mut [*mut L2<W2, T>; W1],
}

impl<const W1: usize, const W2: usize, T: Provenance> L1<W1, W2, T> {
    fn new() -> Self {
        let align = std::mem::align_of::<[*mut L2<W2, T>; W1]>();
        let size = std::mem::size_of::<[*mut L2<W2, T>; W1]>();
        unsafe {
            let entries = alloc(Layout::from_size_align(size, align).unwrap());
            let entries = entries.cast::<[*mut L2<W2, T>; W1]>();
            Self { entries }
        }
    }
    #[inline]
    #[cfg(target_endian = "little")]
    fn lookup(&mut self, index: usize) -> &mut T {
        use std::ops::Shr;
        let as_l1 = index.shr(W2).bitand(W1 - 1);
        let as_l2 = index.bitand(W2 - 1);
        debug_assert!(as_l1 < W1);
        debug_assert!(as_l2 < W2);
        unsafe {
            let l2_entry = &mut *(*self.entries)[as_l1];
            l2_entry.lookup(as_l2)
        }
    }
    #[inline(always)]
    #[cfg(target_endian = "big")]
    fn lookup(&mut index: usize) -> &mut T {
        use std::ops::Shl;
        let as_l1 = index.shr(W2).bitand(W1 - 1);
        let as_l2 = index.bitand(W2 - 1);
        debug_assert!(as_l1 < W1);
        debug_assert!(as_l2 < W2);
        unsafe {
            let l2_entry = &mut *(*self.bytes)[as_l1];
            l2_entry.lookup(as_l2)
        }
    }
}

impl<const W1: usize, const W2: usize, T: Provenance> Drop for L1<W1, W2, T> {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align(W1, 1).unwrap();
            let entries = self.entries.cast::<u8>();
            dealloc(entries, layout);
        }
    }
}

pub struct MemTable<const W1: usize, const W2: usize, T: Provenance> {
    l1: L1<W1, W2, T>,
}

impl<const W1: usize, const W2: usize, T: Provenance> Default for MemTable<W1, W2, T> {
    fn default() -> Self {
        let l1 = L1::<W1, W2, T>::new();
        Self { l1 }
    }
}
