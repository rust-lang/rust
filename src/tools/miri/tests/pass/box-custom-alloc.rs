//@revisions: stack tree tree_implicit_writes
//@[tree_implicit_writes]compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(allocator_api)]

use std::alloc::{Alloc, AllocError, Allocator, Layout};
use std::cell::Cell;
use std::mem::MaybeUninit;
use std::ptr::{NonNull};

struct OnceAlloc<'a> {
    space: Cell<&'a mut [MaybeUninit<u8>]>,
}

unsafe impl<'shared, 'a: 'shared> Alloc for &'shared OnceAlloc<'a> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        let space = self.space.replace(&mut []);

        let (ptr, len) = (space.as_mut_ptr(), space.len());

        if ptr.align_offset(layout.align()) != 0 || len < layout.size() {
            return Err(AllocError);
        }

        unsafe { Ok(NonNull::new_unchecked(ptr as *mut u8)) }
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}

unsafe impl<'shared, 'a: 'shared> Allocator for &'shared OnceAlloc<'a> {
    type Alloc = Self;
    fn alloc_ref(&self) -> &Self::Alloc {
        self
    }
}

trait MyTrait {
    fn hello(&self) -> u8;
}

impl MyTrait for [u8; 1] {
    fn hello(&self) -> u8 {
        self[0]
    }
}

trait TheTrait: MyTrait {}

impl TheTrait for [u8; 1] {}

/// `Box<T, G>` is a `ScalarPair` where the 2nd component is the allocator.
fn test1() {
    let mut space = vec![MaybeUninit::new(0); 1];
    let once_alloc = OnceAlloc { space: Cell::new(&mut space[..]) };

    let boxed = Box::new_in([42u8; 1], &once_alloc);
    let _val = *boxed;
    let with_dyn: Box<dyn TheTrait, &OnceAlloc> = boxed;
    assert_eq!(42, with_dyn.hello());
    let with_dyn: Box<dyn MyTrait, &OnceAlloc> = with_dyn; // upcast
    assert_eq!(42, with_dyn.hello());
}

// Make the allocator itself so big that the Box is not even a ScalarPair any more.
struct OnceAllocRef<'s, 'a>(&'s OnceAlloc<'a>, #[allow(dead_code)] u64);

unsafe impl<'shared, 'a: 'shared> Alloc for OnceAllocRef<'shared, 'a> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        Alloc::allocate(self.0.alloc_ref(), layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        Alloc::deallocate(self.0.alloc_ref(), ptr, layout)
    }
}

unsafe impl<'shared, 'a: 'shared> Allocator for OnceAllocRef<'shared, 'a> {
    type Alloc = Self;
    fn alloc_ref(&self) -> &Self::Alloc {
        self
    }
}

/// `Box<T, G>` is an `Aggregate`.
fn test2() {
    let mut space = vec![MaybeUninit::new(0); 1];
    let once_alloc = OnceAlloc { space: Cell::new(&mut space[..]) };

    let boxed = Box::new_in([42u8; 1], OnceAllocRef(&once_alloc, 0));
    let _val = *boxed;
    let with_dyn: Box<dyn TheTrait, OnceAllocRef> = boxed;
    assert_eq!(42, with_dyn.hello());
    let with_dyn: Box<dyn MyTrait, OnceAllocRef> = with_dyn; // upcast
    assert_eq!(42, with_dyn.hello());
}

fn main() {
    test1();
    test2();
}
