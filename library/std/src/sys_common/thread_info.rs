#![allow(dead_code)] // stack_guard isn't used right now on all platforms

use crate::cell::OnceCell;
use crate::sys;
use crate::sys::thread::guard::Guard;
use crate::thread::Thread;

struct ThreadInfo {
    stack_guard: OnceCell<Guard>,
    thread: OnceCell<Thread>,
}

thread_local! {
   static THREAD_INFO: ThreadInfo = const { ThreadInfo {
       stack_guard: OnceCell::new(),
       thread: OnceCell::new()
   } };
}

impl ThreadInfo {
    fn with<R, F>(f: F) -> Option<R>
    where
        F: FnOnce(&Thread, &OnceCell<Guard>) -> R,
    {
        THREAD_INFO
            .try_with(move |thread_info| {
                let thread =
                    thread_info.thread.get_or_init(|| Thread::new(sys::thread::Thread::get_name()));
                f(thread, &thread_info.stack_guard)
            })
            .ok()
    }
}

pub fn current_thread() -> Option<Thread> {
    ThreadInfo::with(|thread, _| thread.clone())
}

pub fn stack_guard() -> Option<Guard> {
    ThreadInfo::with(|_, guard| guard.get().cloned()).flatten()
}

/// Set new thread info, panicking if it has already been initialized
#[allow(unreachable_code, unreachable_patterns)] // some platforms don't use stack_guard
pub fn set(stack_guard: Option<Guard>, thread: Thread) {
    THREAD_INFO.with(move |thread_info| {
        rtassert!(thread_info.stack_guard.get().is_none() && thread_info.thread.get().is_none());
        if let Some(guard) = stack_guard {
            thread_info.stack_guard.set(guard).unwrap();
        }
        thread_info.thread.set(thread).unwrap();
    });
}
