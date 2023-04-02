#![allow(dead_code)] // stack_guard isn't used right now on all platforms

use crate::cell::Cell;
use crate::sys::thread::guard::Guard;
use crate::thread::Thread;

thread_local! {
    static THREAD: Cell<Option<Thread>> = const { Cell::new(None) };
    // Use a separate thread local for the stack guard page location.
    // Since `Guard` does not implement drop, this is always available
    // on systems with ELF-TLS, in particular during TLS destruction.
    static STACK_GUARD: Cell<Option<Guard>> = const { Cell::new(None) };
}

pub fn current_thread() -> Option<Thread> {
    THREAD
        .try_with(|thread| {
            let t = thread.take().unwrap_or_else(|| Thread::new(None));
            let t2 = t.clone();
            thread.set(Some(t));
            t2
        })
        .ok()
}

pub fn stack_guard() -> Option<Guard> {
    STACK_GUARD
        .try_with(|guard| {
            let g = guard.take();
            let g2 = g.clone();
            guard.set(g);
            g2
        })
        .ok()
        .flatten()
}

pub fn set(stack_guard: Option<Guard>, thread: Thread) {
    rtassert!(STACK_GUARD.replace(stack_guard).is_none());
    rtassert!(THREAD.replace(Some(thread)).is_none());
}
