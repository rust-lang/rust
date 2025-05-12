//! Regression test for <https://github.com/rust-lang/miri/issues/3341>:
//! If `Box` has a local allocator, then it can't be `noalias` as the allocator
//! may want to access allocator state based on the data pointer.

//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(allocator_api)]

use std::alloc::{AllocError, Allocator, Layout};
use std::cell::{Cell, UnsafeCell};
use std::mem;
use std::ptr::{self, NonNull, addr_of};
use std::thread::{self, ThreadId};

const BIN_SIZE: usize = 8;

// A bin represents a collection of blocks of a specific layout.
#[repr(align(128))]
struct MyBin {
    top: Cell<usize>,
    thread_id: ThreadId,
    memory: UnsafeCell<[usize; BIN_SIZE]>,
}

impl MyBin {
    fn pop(&self) -> Option<NonNull<u8>> {
        let top = self.top.get();
        if top == BIN_SIZE {
            return None;
        }
        // Cast the *entire* thing to a raw pointer to not restrict its provenance.
        let bin = self as *const MyBin;
        let base_ptr = UnsafeCell::raw_get(unsafe { addr_of!((*bin).memory) }).cast::<usize>();
        let ptr = unsafe { NonNull::new_unchecked(base_ptr.add(top)) };
        self.top.set(top + 1);
        Some(ptr.cast())
    }

    // Pretends to not be a throwaway allocation method like this. A more realistic
    // substitute is using intrusive linked lists, which requires access to the
    // metadata of this bin as well.
    unsafe fn push(&self, ptr: NonNull<u8>) {
        // For now just check that this really is in this bin.
        let start = self.memory.get().addr();
        let end = start + BIN_SIZE * mem::size_of::<usize>();
        let addr = ptr.addr().get();
        assert!((start..end).contains(&addr));
    }
}

// A collection of bins.
struct MyAllocator {
    thread_id: ThreadId,
    // Pretends to be some complex collection of bins, such as an array of linked lists.
    bins: Box<[MyBin; 1]>,
}

impl MyAllocator {
    fn new() -> Self {
        let thread_id = thread::current().id();
        MyAllocator {
            thread_id,
            bins: Box::new(
                [MyBin { top: Cell::new(0), thread_id, memory: UnsafeCell::default() }; 1],
            ),
        }
    }

    // Pretends to be expensive finding a suitable bin for the layout.
    fn find_bin(&self, layout: Layout) -> Option<&MyBin> {
        if layout == Layout::new::<usize>() { Some(&self.bins[0]) } else { None }
    }
}

unsafe impl Allocator for MyAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // Expensive bin search.
        let bin = self.find_bin(layout).ok_or(AllocError)?;
        let ptr = bin.pop().ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        // Make sure accesses via `self` don't disturb anything.
        let _val = self.bins[0].top.get();
        // Since manually finding the corresponding bin of `ptr` is very expensive,
        // doing pointer arithmetics is preferred.
        // But this means we access `top` via `ptr` rather than `self`!
        // That is fundamentally the source of the aliasing trouble in this example.
        let their_bin = ptr.as_ptr().map_addr(|addr| addr & !127).cast::<MyBin>();
        let thread_id = ptr::read(ptr::addr_of!((*their_bin).thread_id));
        if self.thread_id == thread_id {
            unsafe { (*their_bin).push(ptr) };
        } else {
            todo!("Deallocating from another thread");
        }
        // Make sure we can also still access this via `self` after the rest is done.
        let _val = self.bins[0].top.get();
    }
}

// Make sure to involve `Box` in allocating these,
// as that's where `noalias` may come from.
fn v1<T, A: Allocator>(t: T, a: A) -> Vec<T, A> {
    (Box::new_in([t], a) as Box<[T], A>).into_vec()
}
fn v2<T, A: Allocator>(t: T, a: A) -> Vec<T, A> {
    let v = v1(t, a);
    // There was a bug in `into_boxed_slice` that caused aliasing issues,
    // so round-trip through that as well.
    v.into_boxed_slice().into_vec()
}

fn main() {
    assert!(mem::size_of::<MyBin>() <= 128); // if it grows bigger, the trick to access the "header" no longer works
    let my_alloc = MyAllocator::new();
    let a = v1(1usize, &my_alloc);
    let b = v2(2usize, &my_alloc);
    assert_eq!(a[0] + 1, b[0]);
    assert_eq!(addr_of!(a[0]).wrapping_add(1), addr_of!(b[0]));
    drop((a, b));
}
