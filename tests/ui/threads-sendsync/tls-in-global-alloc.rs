//@ run-pass
//@ needs-threads

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static GLOBAL: AtomicUsize = AtomicUsize::new(0);

struct Local;

thread_local! {
    static LOCAL: Local = {
        GLOBAL.fetch_or(1, Ordering::Relaxed);
        Local
    };
}

impl Drop for Local {
    fn drop(&mut self) {
        GLOBAL.fetch_or(2, Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOC: Alloc = Alloc;
struct Alloc;

unsafe impl GlobalAlloc for Alloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LOCAL.with(|_local| {});
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LOCAL.with(|_local| {});
        unsafe { System.dealloc(ptr, layout) }
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
