//! This test represents a core part of `SmallVec`'s `extend_impl`.
//! What makes it interesting as a test is that it relies on Stacked Borrow's "quirk"
//! in a fundamental, hard-to-fix-without-full-trees way.

//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

use std::marker::PhantomData;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ptr::NonNull;

#[repr(C)]
pub union RawSmallVec<T, const N: usize> {
    inline: ManuallyDrop<MaybeUninit<[T; N]>>,
    heap: (NonNull<T>, usize),
}

impl<T, const N: usize> RawSmallVec<T, N> {
    const fn new() -> Self {
        Self::new_inline(MaybeUninit::uninit())
    }

    const fn new_inline(inline: MaybeUninit<[T; N]>) -> Self {
        Self { inline: ManuallyDrop::new(inline) }
    }

    const fn as_mut_ptr_inline(&mut self) -> *mut T {
        (unsafe { &raw mut self.inline }) as *mut T
    }

    const unsafe fn as_mut_ptr_heap(&mut self) -> *mut T {
        self.heap.0.as_ptr()
    }
}

#[repr(transparent)]
#[derive(Clone, Copy)]
struct TaggedLen(usize);

impl TaggedLen {
    pub const fn new(len: usize, on_heap: bool, is_zst: bool) -> Self {
        if is_zst {
            debug_assert!(!on_heap);
            TaggedLen(len)
        } else {
            debug_assert!(len < isize::MAX as usize);
            TaggedLen((len << 1) | on_heap as usize)
        }
    }

    pub const fn on_heap(self, is_zst: bool) -> bool {
        if is_zst { false } else { (self.0 & 1_usize) == 1 }
    }

    pub const fn value(self, is_zst: bool) -> usize {
        if is_zst { self.0 } else { self.0 >> 1 }
    }
}

#[repr(C)]
pub struct SmallVec<T, const N: usize> {
    len: TaggedLen,
    raw: RawSmallVec<T, N>,
    _marker: PhantomData<T>,
}

impl<T, const N: usize> SmallVec<T, N> {
    pub const fn new() -> SmallVec<T, N> {
        Self {
            len: TaggedLen::new(0, false, Self::is_zst()),
            raw: RawSmallVec::new(),
            _marker: PhantomData,
        }
    }

    const fn is_zst() -> bool {
        size_of::<T>() == 0
    }

    pub const fn as_mut_ptr(&mut self) -> *mut T {
        if self.len.on_heap(Self::is_zst()) {
            // SAFETY: see above
            unsafe { self.raw.as_mut_ptr_heap() }
        } else {
            self.raw.as_mut_ptr_inline()
        }
    }

    pub const fn len(&self) -> usize {
        self.len.value(Self::is_zst())
    }
}

fn main() {
    let mut v = SmallVec::<i32, 4>::new();
    let ptr = v.as_mut_ptr();
    let _len = v.len(); // this call incurs a reborrow which just barely does not invalidate `ptr`
    unsafe { ptr.write(0) };
}
