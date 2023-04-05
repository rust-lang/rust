#![allow(dead_code)] // stack_guard isn't used right now on all platforms

use crate::cell::OnceCell;
use crate::sys::thread::guard::Guard;
use crate::thread::Thread;

thread_local! {
    static THREAD: OnceCell<Thread> = const { OnceCell::new() };
    // Use a separate thread local for the stack guard page location.
    // Since `Guard` does not implement drop, this is always available
    // on systems with ELF-TLS, in particular during TLS destruction.
    static STACK_GUARD: OnceCell<Guard> = const { OnceCell::new() };
}

pub fn current_thread() -> Option<Thread> {
    THREAD.try_with(|thread| thread.get_or_init(|| Thread::new(None)).clone()).ok()
}

pub fn stack_guard() -> Option<Guard> {
    STACK_GUARD.try_with(|guard| guard.get().cloned()).ok().flatten()
}

pub fn set(stack_guard: Option<Guard>, thread: Thread) {
    #[allow(unreachable_patterns, unreachable_code)] // On some platforms, `Guard` is `!`.
    if let Some(guard) = stack_guard {
        rtassert!(STACK_GUARD.with(|s| s.set(guard)).is_ok());
    }
    rtassert!(THREAD.with(|t| t.set(thread)).is_ok());
}
