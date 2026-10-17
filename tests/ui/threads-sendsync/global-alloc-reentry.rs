//! test for <https://github.com/rust-lang/rust/issues/160776>
//@ run-pass
//@ needs-threads
//@ only-unix because we use pthread

#![feature(rustc_private)]
extern crate libc;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::Thread;
use std::{mem, ptr};

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PTHREAD_ACTIVE: AtomicBool = AtomicBool::new(false);
static PTHREAD_SUCCEEDED_ALLOC: AtomicBool = AtomicBool::new(false);
static PTHREAD_SUCCEEDED_DEALLOC: AtomicBool = AtomicBool::new(false);

#[global_allocator]
static ALLOC: Alloc = Alloc;
struct Alloc;

fn call_function(f: impl FnOnce()) {
    ALLOCATED.store(0, Ordering::Relaxed);
    f();
    assert_eq!(ALLOCATED.load(Ordering::Relaxed), 0);
}

// https://doc.rust-lang.org/nightly/std/alloc/trait.GlobalAlloc.html#re-entrance
fn guarantee_functions() {
    call_function(|| {
        drop(std::thread::current());
    });

    call_function(|| {
        std::thread::current().unpark();
    });

    call_function(std::thread::park);

    call_function(|| {
        drop(std::thread::current().clone());
    });

    call_function(|| {
        LOCAL_FOR_ALLOCATOR_WITH_DROP
            .with(|local| assert!(local.0.id() == std::thread::current().id()))
    });
}

extern "C" fn start(c: *mut libc::c_void) -> *mut libc::c_void {
    PTHREAD_ACTIVE.swap(true, Ordering::Relaxed);
    std::hint::black_box(vec![1, 2]);

    guarantee_functions();

    unsafe {
        let c: *mut u32 = c.cast();
        drop(Box::from_raw(c))
    }

    PTHREAD_ACTIVE.swap(false, Ordering::Relaxed);
    ptr::null_mut()
}

struct LocalForAllocatorWithDrop(Thread);

thread_local! {
    static LOCAL_FOR_ALLOCATOR_WITH_DROP: LocalForAllocatorWithDrop = {
        LocalForAllocatorWithDrop(std::thread::current())
    }
}

unsafe impl GlobalAlloc for Alloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if PTHREAD_ACTIVE.load(Ordering::Relaxed) {
            PTHREAD_SUCCEEDED_ALLOC.swap(true, Ordering::Relaxed);
        }

        let ret = unsafe { System.alloc(layout) };
        if !ret.is_null() {
            ALLOCATED.fetch_add(1, Ordering::Relaxed);
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if PTHREAD_ACTIVE.load(Ordering::Relaxed) {
            PTHREAD_SUCCEEDED_DEALLOC.swap(true, Ordering::Relaxed);
        }

        unsafe {
            System.dealloc(ptr, layout);
        }
        ALLOCATED.fetch_add(1, Ordering::Relaxed);
    }
}

fn main() {
    unsafe {
        let c: *mut u32 = Box::into_raw(Box::new(1));
        let mut t: libc::pthread_t = mem::zeroed();
        libc::pthread_create(&mut t, ptr::null(), start, c.cast());
        libc::pthread_join(t, ptr::null_mut());
    }

    // I mostly did this because I wanted to see if the allocator would run in pthreads
    assert!(PTHREAD_SUCCEEDED_ALLOC.load(Ordering::Relaxed));
    assert!(PTHREAD_SUCCEEDED_DEALLOC.load(Ordering::Relaxed));

    std::thread::spawn(|| {
        std::hint::black_box(vec![1, 2]);
        guarantee_functions();
    })
    .join()
    .unwrap();

    guarantee_functions();
}
