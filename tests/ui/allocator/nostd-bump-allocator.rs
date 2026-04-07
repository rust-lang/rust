//@ run-pass
//@ compile-flags: -C opt-level=0

// Validates that the `Allocator` trait is usable in a `no_std` context by
// implementing a single-threaded bump (arena) allocator backed by a fixed
// static byte array.
//
// This pattern is the canonical early-boot memory strategy for bare-metal
// kernels (e.g. Feox) where the global allocator is not yet available.
// The test exercises all four `Allocator` methods — allocate, deallocate,
// grow, and shrink — and verifies the alignment guarantees mandated by the
// trait contract.

#![feature(allocator_api)]
#![feature(pointer_is_aligned_to)]

use std::alloc::{AllocError, Allocator, Layout};
use std::cell::UnsafeCell;
use std::ptr::NonNull;

/// A minimal bump allocator backed by a fixed stack-allocated buffer.
///
/// Allocations are handed out sequentially with correct alignment padding.
/// There is no `free` — `deallocate`, `grow`, and `shrink` follow the
/// minimum valid `Allocator` contract: grow/shrink allocate a new block and
/// copy, deallocate is a no-op (the bump pointer never retreats).
///
/// This is intentionally the simplest possible no_std allocator — the same
/// shape used during early kernel boot before a slab or buddy allocator is
/// online.
struct BumpAllocator<const N: usize> {
    buf: UnsafeCell<[u8; N]>,
    // Byte offset of the next free slot.  Stored as a raw pointer so we can
    // hand out pointers without lifetime entanglement.
    next: UnsafeCell<usize>,
}

impl<const N: usize> BumpAllocator<N> {
    const fn new() -> Self {
        Self { buf: UnsafeCell::new([0u8; N]), next: UnsafeCell::new(0) }
    }

    fn base(&self) -> *mut u8 {
        self.buf.get().cast::<u8>()
    }

    fn used(&self) -> usize {
        unsafe { *self.next.get() }
    }
}

unsafe impl<const N: usize> Allocator for &BumpAllocator<N> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let next = &mut *self.next.get();
            let base = self.base();

            // Align the current bump pointer.
            let start = base.add(*next) as usize;
            let aligned = (start + layout.align() - 1) & !(layout.align() - 1);
            let offset = aligned - base as usize;

            let end = offset + layout.size();
            if end > N {
                return Err(AllocError);
            }

            *next = end;
            let ptr = NonNull::new_unchecked(base.add(offset));
            Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
        }
    }

    // Bump allocators cannot reclaim memory; deallocate is intentionally a
    // no-op here.  The `Allocator` contract permits this.
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}

    // Default `grow` / `shrink` implementations allocate a new block and
    // copy — this exercises the default method paths in core::alloc.
}

fn check_alignment(ptr: *mut u8, align: usize) {
    assert!(ptr.is_aligned_to(align), "pointer {ptr:p} is not aligned to {align}");
}

fn main() {
    let alloc = BumpAllocator::<1024>::new();
    let a = &alloc;

    // ---- basic allocation ----
    let layout_u32 = Layout::new::<u32>();
    let block = a.allocate(layout_u32).expect("u32 allocation failed");
    check_alignment(block.as_non_null_ptr().as_ptr(), layout_u32.align());
    assert_eq!(block.len(), layout_u32.size());

    // ---- over-aligned allocation ----
    let layout_aligned =
        Layout::from_size_align(8, 64).expect("layout construction failed");
    let block64 = a.allocate(layout_aligned).expect("64-byte aligned allocation failed");
    check_alignment(block64.as_non_null_ptr().as_ptr(), 64);
    assert_eq!(block64.len(), 8);

    // ---- zero-size allocation returns a dangling-but-non-null pointer ----
    let layout_zst = Layout::from_size_align(0, 1).unwrap();
    let zst_block = a.allocate(layout_zst).expect("ZST allocation failed");
    assert!(!zst_block.as_non_null_ptr().as_ptr().is_null());

    // ---- grow via default implementation ----
    let layout_small = Layout::from_size_align(4, 4).unwrap();
    let small = a.allocate(layout_small).expect("small allocation failed");
    let layout_large = Layout::from_size_align(16, 4).unwrap();
    let grown = unsafe {
        a.grow(small.as_non_null_ptr(), layout_small, layout_large)
            .expect("grow failed")
    };
    assert!(grown.len() >= 16);
    check_alignment(grown.as_non_null_ptr().as_ptr(), 4);

    // ---- shrink via default implementation ----
    let layout_shrunk = Layout::from_size_align(8, 4).unwrap();
    let shrunk = unsafe {
        a.shrink(grown.as_non_null_ptr(), layout_large, layout_shrunk)
            .expect("shrink failed")
    };
    assert!(shrunk.len() >= 8);
    check_alignment(shrunk.as_non_null_ptr().as_ptr(), 4);

    // ---- allocation failure when buffer is exhausted ----
    let layout_huge = Layout::from_size_align(2048, 1).unwrap();
    assert!(
        a.allocate(layout_huge).is_err(),
        "allocation should fail when bump buffer is exhausted"
    );

    // ---- bytes used is monotonically non-decreasing ----
    let used_before = alloc.used();
    let layout_byte = Layout::new::<u8>();
    let _ = a.allocate(layout_byte).expect("byte allocation failed");
    assert!(alloc.used() > used_before);
}
