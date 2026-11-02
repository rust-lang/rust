//@revisions: stack tree tree_implicit_writes
//@[tree_implicit_writes]compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes
//@[tree]compile-flags: -Zmiri-tree-borrows

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::cell::Cell;
use std::mem::MaybeUninit;
use std::ptr::{self, NonNull};

struct OnceAlloc<'a> {
    space: Cell<&'a mut [MaybeUninit<u8>]>,
}

unsafe impl<'shared, 'a: 'shared> Allocator for &'shared OnceAlloc<'a> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let space = self.space.replace(&mut []);

        let (ptr, len) = (space.as_mut_ptr(), space.len());

        if ptr.align_offset(layout.align()) != 0 || len < layout.size() {
            return Err(AllocError);
        }

        let slice_ptr = ptr::slice_from_raw_parts_mut(ptr as *mut u8, len);
        unsafe { Ok(NonNull::new_unchecked(slice_ptr)) }
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
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

unsafe impl<'shared, 'a: 'shared> Allocator for OnceAllocRef<'shared, 'a> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.0.deallocate(ptr, layout)
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

fn test3() {
    use std::ptr::{NonNull, slice_from_raw_parts_mut};
    use std::sync::atomic::{AtomicBool, Ordering};

    static mut ALLOCATION: usize = 0;
    static ALLOCATED: AtomicBool = AtomicBool::new(false);

    #[derive(Clone, Copy)]
    struct A;
    unsafe impl Allocator for A {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            if layout != Layout::new::<usize>() {
                return Err(AllocError);
            }
            if ALLOCATED.swap(true, Ordering::Acquire) {
                return Err(AllocError);
            }
            NonNull::new(slice_from_raw_parts_mut(&raw mut ALLOCATION as *mut u8, 8))
                .ok_or(AllocError)
        }
        unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
            ALLOCATED.store(false, Ordering::Release);
        }
    }

    fn foo<A: Allocator>(a: A, b1: Box<usize, A>) {
        assert_eq!(*b1, 1);
        drop(b1);
        let b3 = Box::new_in(3usize, a);
        assert_eq!(*b3, 3);
    }

    let b1 = Box::new_in(1usize, A);
    foo(A, b1);
}

fn test_into_raw() {
    struct MyMetadataAlloc;

    fn widen(layout: Layout) -> Layout {
        Layout::from_size_align(layout.size() + 10, layout.align())
            .unwrap_or_else(|_| std::process::abort())
    }
    unsafe impl Allocator for MyMetadataAlloc {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            let ptr = Global.allocate(widen(layout))?;

            // store some important metadata
            unsafe { ptr.cast::<u8>().add(layout.size() + 5).write(42) }

            Ok(NonNull::slice_from_raw_parts(ptr.cast(), layout.size()))
        }
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            // SAFETY: we get back a pointer with the same provenance as
            // returned by `allocate`, valid over the full `size() + 10` bytes.
            unsafe {
                assert_eq!(ptr.add(layout.size() + 5).replace(1), 42);
                Global.deallocate(ptr, widen(layout))
            }
        }
    }

    unsafe {
        let b = Box::new_in(1i32, MyMetadataAlloc);

        let (r, alloc) = Box::into_raw_with_allocator(b);

        // ptr is valid
        assert_eq!(std::mem::replace(&mut *r, 2), 1);

        // ptr can be deallocated again
        drop(Box::from_raw_in(r, alloc));
    }
}

fn main() {
    test1();
    test2();
    test3();
    test_into_raw();
}
