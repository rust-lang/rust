//@ run-pass
//@ needs-threads

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{Thread, ThreadId};

static GLOBAL: AtomicUsize = AtomicUsize::new(0);
static SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS: AtomicBool = AtomicBool::new(false);
static LOCAL_TRY_WITH_SUCCEEDED_ALLOC: AtomicBool = AtomicBool::new(false);
static LOCAL_TRY_WITH_SUCCEEDED_DEALLOC: AtomicBool = AtomicBool::new(false);

struct LocalForAllocatorWithoutDrop(ThreadId);

struct LocalForAllocatorWithDrop(Thread);

impl Drop for LocalForAllocatorWithDrop {
    fn drop(&mut self) {
        GLOBAL.fetch_or(2, Ordering::Relaxed);
    }
}

struct LocalForUser(u32);

impl Drop for LocalForUser {
    // A user might call the global allocator in a thread-local drop.
    fn drop(&mut self) {
        self.0 += 1;
        drop(black_box(Box::new(self.0)))
    }
}

thread_local! {
    static LOCAL_FOR_USER0: LocalForUser = LocalForUser(0);
    static LOCAL_FOR_ALLOCATOR_WITHOUT_DROP: LocalForAllocatorWithoutDrop = {
        LocalForAllocatorWithoutDrop(std::thread::current().id())
    };
    static LOCAL_FOR_ALLOCATOR_WITH_DROP: LocalForAllocatorWithDrop = {
        GLOBAL.fetch_or(1, Ordering::Relaxed);
        LocalForAllocatorWithDrop(std::thread::current())
    };
    static LOCAL_FOR_USER1: LocalForUser = LocalForUser(1);
}

#[global_allocator]
static ALLOC: Alloc = Alloc;
struct Alloc;

unsafe impl GlobalAlloc for Alloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Make sure we aren't re-entrant.
        assert!(!SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.load(Ordering::Relaxed));
        SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.store(true, Ordering::Relaxed);

        // Should be infallible.
        LOCAL_FOR_ALLOCATOR_WITHOUT_DROP.with(|local| {
            assert!(local.0 == std::thread::current().id());
        });

        // May fail once thread-local destructors start running, and ours has
        // been ran.
        let try_with_ret = LOCAL_FOR_ALLOCATOR_WITH_DROP.try_with(|local| {
            assert!(local.0.id() == std::thread::current().id());
        });
        LOCAL_TRY_WITH_SUCCEEDED_ALLOC.fetch_or(try_with_ret.is_ok(), Ordering::Relaxed);

        let ret = unsafe { System.alloc(layout) };
        SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.store(false, Ordering::Relaxed);
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Make sure we aren't re-entrant.
        assert!(!SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.load(Ordering::Relaxed));
        SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.store(true, Ordering::Relaxed);

        // Should be infallible.
        LOCAL_FOR_ALLOCATOR_WITHOUT_DROP.with(|local| {
            assert!(local.0 == std::thread::current().id());
        });

        // May fail once thread-local destructors start running, and ours has
        // been ran.
        let try_with_ret = LOCAL_FOR_ALLOCATOR_WITH_DROP.try_with(|local| {
            assert!(local.0.id() == std::thread::current().id());
        });
        LOCAL_TRY_WITH_SUCCEEDED_DEALLOC.fetch_or(try_with_ret.is_ok(), Ordering::Relaxed);

        unsafe { System.dealloc(ptr, layout) }
        SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.store(false, Ordering::Relaxed);
    }
}

fn main() {
    std::thread::spawn(|| {
        LOCAL_FOR_USER0.with(|l| assert!(l.0 == 0));
        std::hint::black_box(vec![1, 2]);
        assert!(GLOBAL.load(Ordering::Relaxed) == 1);
        LOCAL_FOR_USER1.with(|l| assert!(l.0 == 1));
    })
    .join()
    .unwrap();
    assert!(GLOBAL.load(Ordering::Relaxed) == 3);
    assert!(LOCAL_TRY_WITH_SUCCEEDED_ALLOC.load(Ordering::Relaxed));
    assert!(LOCAL_TRY_WITH_SUCCEEDED_DEALLOC.load(Ordering::Relaxed));
}
