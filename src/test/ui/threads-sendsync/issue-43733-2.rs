// dont-check-compiler-stderr

#![feature(cfg_target_thread_local, thread_local_internals)]

// On platforms *without* `#[thread_local]`, use
// a custom non-`Sync` type to fake the same error.
#[cfg(not(target_thread_local))]
struct Key<T> {
    _data: std::cell::UnsafeCell<Option<T>>,
    _flag: std::cell::Cell<()>,
}

#[cfg(not(target_thread_local))]
impl<T> Key<T> {
    const fn new() -> Self {
        Key {
            _data: std::cell::UnsafeCell::new(None),
            _flag: std::cell::Cell::new(()),
        }
    }
}

#[cfg(target_thread_local)]
use std::thread::__FastLocalKeyInner as Key;

static __KEY: Key<()> = Key::new();
//~^ ERROR `UnsafeCell<Option<()>>` cannot be shared between threads
//~| ERROR cannot be shared between threads safely [E0277]

fn main() {}
