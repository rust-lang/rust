// https://github.com/rust-lang/rust/issues/159923
//@ edition:2024
//@ run-pass
//@ needs-threads
//@ needs-unwind

#![feature(alloc_error_hook, thread_spawn_hook)]

use std::{
    alloc::{GlobalAlloc, Layout, System, set_alloc_error_hook},
    sync::atomic::{AtomicBool, AtomicU32, Ordering},
    thread,
};

struct FailingGlobalAlloc;
unsafe impl GlobalAlloc for FailingGlobalAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if FAIL_NEXT_ALLOC.swap(false, Ordering::Relaxed) {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOC: FailingGlobalAlloc = FailingGlobalAlloc;

static FAIL_NEXT_ALLOC: AtomicBool = AtomicBool::new(false);

fn spawn_and_join() {
    thread::scope(|scope| {
        scope.spawn(|| {});
    });
}

fn main() {
    static COUNT: AtomicU32 = AtomicU32::new(0);

    thread::add_spawn_hook(|_| || _ = COUNT.fetch_add(1, Ordering::Relaxed));
    thread::add_spawn_hook(|_| || _ = COUNT.fetch_add(1, Ordering::Relaxed));

    spawn_and_join();
    spawn_and_join();
    assert_eq!(COUNT.swap(0, Ordering::Relaxed), 4);

    set_alloc_error_hook(|_| {
        spawn_and_join();
        assert_eq!(COUNT.swap(0, Ordering::Relaxed), 2);
        panic!()
    });

    FAIL_NEXT_ALLOC.store(true, Ordering::Relaxed);
    std::panic::catch_unwind(|| {
        std::thread::add_spawn_hook(|_| || {});
    })
    .unwrap_err();

    spawn_and_join();
    assert_eq!(COUNT.swap(0, Ordering::Relaxed), 2);
}
