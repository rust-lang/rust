//@ run-pass
//@ needs-threads

use std::alloc::{GlobalAlloc, Layout, System};
use std::thread::Thread;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};

static GLOBAL: AtomicUsize = AtomicUsize::new(0);

struct Local(Thread);

thread_local! {
    static LOCAL: Local = {
        GLOBAL.fetch_or(1, Ordering::Relaxed);
        Local(std::thread::current())
    };
}

impl Drop for Local {
    fn drop(&mut self) {
        GLOBAL.fetch_or(2, Ordering::Relaxed);
    }
}

static SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS: AtomicBool = AtomicBool::new(false);

#[global_allocator]
static ALLOC: Alloc = Alloc;
struct Alloc;

unsafe impl GlobalAlloc for Alloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Make sure we aren't re-entrant.
        assert!(!SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.load(Ordering::Relaxed));
        SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.store(true, Ordering::Relaxed);
        LOCAL.with(|local| {
            assert!(local.0.id() == std::thread::current().id());
        });
        let ret = unsafe { System.alloc(layout) };
        SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.store(false, Ordering::Relaxed);
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Make sure we aren't re-entrant.
        assert!(!SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.load(Ordering::Relaxed));
        SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.store(true, Ordering::Relaxed);
        LOCAL.with(|local| {
            assert!(local.0.id() == std::thread::current().id());
        });
        unsafe { System.dealloc(ptr, layout) }
        SHOULD_PANIC_ON_GLOBAL_ALLOC_ACCESS.store(false, Ordering::Relaxed);
    }
}

fn main() {
    std::thread::spawn(|| {
        std::hint::black_box(vec![1, 2]);
        assert!(GLOBAL.load(Ordering::Relaxed) == 1);
    })
    .join()
    .unwrap();
    assert!(GLOBAL.load(Ordering::Relaxed) == 3);
}
