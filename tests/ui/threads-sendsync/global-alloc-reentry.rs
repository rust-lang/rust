//! test for <https://github.com/rust-lang/rust/issues/160776>
//@ run-pass
//@ needs-threads
//@ only-unix because we use pthread

#![feature(rustc_private)]
extern crate libc;

use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr;
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::Thread;

static SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS: AtomicBool = AtomicBool::new(false);
static PTHREAD_ACTIVE: AtomicBool = AtomicBool::new(false);
static PTHREAD_SUCCEEDED_ALLOC: AtomicBool = AtomicBool::new(false);
static PTHREAD_SUCCEEDED_DEALLOC: AtomicBool = AtomicBool::new(false);
static LOCAL_TRY_WITH_SUCCEEDED_ALLOC: AtomicBool = AtomicBool::new(false);
static LOCAL_TRY_WITH_SUCCEEDED_DEALLOC: AtomicBool = AtomicBool::new(false);


#[global_allocator]
static ALLOC: Alloc = Alloc;
struct Alloc;

extern "C" fn start(c: *mut libc::c_void) -> *mut libc::c_void {
    PTHREAD_ACTIVE.store(true, Ordering::Relaxed);
    std::hint::black_box(vec![1, 2]);
    unsafe {
        let c: *mut u32 = c.cast();
        drop(Box::from_raw(c))
    }

    PTHREAD_ACTIVE.store(false, Ordering::Relaxed);
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
        assert!(!SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.load(Ordering::Relaxed));
        SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.store(true, Ordering::Relaxed);

        if PTHREAD_ACTIVE.load(Ordering::Relaxed) {
            PTHREAD_SUCCEEDED_ALLOC.store(true, Ordering::Relaxed);
        }

        // https://doc.rust-lang.org/nightly/std/alloc/trait.GlobalAlloc.html#re-entrance
        let th : Thread = std::thread::current();
        th.unpark();
        std::thread::park();
        drop(th.clone());

        let try_with_ret = LOCAL_FOR_ALLOCATOR_WITH_DROP.try_with(|local| {
            assert!(local.0.id() == std::thread::current().id());
        });
        LOCAL_TRY_WITH_SUCCEEDED_ALLOC.fetch_or(try_with_ret.is_ok(), Ordering::Relaxed);

        let ret = unsafe {
            System.alloc(layout)
        };
        SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.store(false, Ordering::Relaxed);
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        assert!(!SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.load(Ordering::Relaxed));
        SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.store(true, Ordering::Relaxed);

        if PTHREAD_ACTIVE.load(Ordering::Relaxed) {
            PTHREAD_SUCCEEDED_DEALLOC.store(true, Ordering::Relaxed);
        }

        let th : Thread = std::thread::current();
        th.unpark();
        std::thread::park();
        drop(th.clone());

        let try_with_ret = LOCAL_FOR_ALLOCATOR_WITH_DROP.try_with(|local| {
            assert!(local.0.id() == std::thread::current().id());
        });
        LOCAL_TRY_WITH_SUCCEEDED_DEALLOC.fetch_or(try_with_ret.is_ok(), Ordering::Relaxed);

        SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.store(false, Ordering::Relaxed);
        unsafe {
            System.dealloc(ptr, layout);
        }
    }
}


fn main() {
    unsafe {
        let c : *mut u32 = Box::into_raw(Box::new(1));
        let mut t : libc::pthread_t = mem::zeroed();
        libc::pthread_create(&mut t, ptr::null(), start, c.cast());
        libc::pthread_join(t, ptr::null_mut());
    }

    // I mostly did this because I wanted to see if the allocator would run in pthreads
    assert!(PTHREAD_SUCCEEDED_ALLOC.load(Ordering::Relaxed));
    assert!(PTHREAD_SUCCEEDED_DEALLOC.load(Ordering::Relaxed));

    assert!(LOCAL_TRY_WITH_SUCCEEDED_ALLOC.load(Ordering::Relaxed));
    assert!(LOCAL_TRY_WITH_SUCCEEDED_DEALLOC.load(Ordering::Relaxed));

    LOCAL_TRY_WITH_SUCCEEDED_ALLOC.store(false, Ordering::Relaxed);
    LOCAL_TRY_WITH_SUCCEEDED_DEALLOC.store(false, Ordering::Relaxed);

    std::thread::spawn(|| {
        std::hint::black_box(vec![1, 2]);
    })
    .join()
    .unwrap();

    assert!(LOCAL_TRY_WITH_SUCCEEDED_ALLOC.load(Ordering::Relaxed));
    assert!(LOCAL_TRY_WITH_SUCCEEDED_DEALLOC.load(Ordering::Relaxed));
}
