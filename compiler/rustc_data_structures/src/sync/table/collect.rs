//! An API for quiescent state based reclamation.

use crate::{scopeguard::guard, util::cold_path};
use parking_lot::Mutex;
use std::arch::asm;
use std::{
    cell::Cell,
    collections::HashMap,
    intrinsics::unlikely,
    marker::PhantomData,
    mem,
    sync::LazyLock,
    sync::atomic::{AtomicUsize, Ordering},
    thread::{self, ThreadId},
};

mod code;

// TODO: Use a reference count of pins and a PinRef ZST with a destructor that drops the reference
// so a closure with the `pin` function isn't required.

static EVENTS: AtomicUsize = AtomicUsize::new(0);

/// Represents a proof that no deferred methods will run for the lifetime `'a`.
/// It can be used to access data structures in a lock-free manner.
#[derive(Clone, Copy)]
pub struct Pin<'a> {
    _private: PhantomData<&'a ()>,
}

// FIXME: Prevent pin calls inside the callback?
/// This schedules a closure to run at some point after all threads are outside their current pinned
/// regions.
///
/// The closure will be called by the [collect] method.
///
/// # Safety
/// This method is unsafe since the closure is not required to be `'static`.
/// It's up to the caller to ensure the closure does not access freed memory.
/// A `move` closure is recommended to avoid accidental references to stack variables.
pub unsafe fn defer_unchecked<F>(f: F)
where
    F: FnOnce(),
    F: Send,
{
    unsafe {
        let f: Box<dyn FnOnce() + Send> = Box::new(f);
        let f: Box<dyn FnOnce() + Send + 'static> = mem::transmute(f);

        COLLECTOR.lock().defer(f);

        EVENTS.fetch_add(1, Ordering::Release);
    }
}

#[thread_local]
static DATA: Data = Data {
    pinned: Cell::new(false),
    registered: Cell::new(false),
    seen_events: Cell::new(0),
};

struct Data {
    pinned: Cell<bool>,
    registered: Cell<bool>,
    seen_events: Cell<usize>,
}

#[inline(never)]
#[cold]
fn panic_pinned() {
    panic!("The current thread was pinned");
}

impl Data {
    #[inline]
    fn assert_unpinned(&self) {
        if self.pinned.get() {
            panic_pinned()
        }
    }

    #[inline(never)]
    #[cold]
    fn register(&self) {
        COLLECTOR.lock().register();
        self.registered.set(true);
    }
}

cfg_if! {
    if #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(miri)
    ))] {
        #[inline]
        fn hide(mut data: *const Data) -> *const Data {
            // Hide the `data` value from LLVM to prevent it from generating multiple TLS accesses
            unsafe {
                asm!("/* {} */", inout(reg) data, options(pure, nomem, nostack, preserves_flags));
            }

            data
        }
    } else {
        #[inline]
        fn hide(data: *const Data) -> *const Data {
            data
        }
    }
}

// Never inline due to thread_local bugs
#[inline(never)]
fn data() -> *const Data {
    &DATA as *const Data
}

// Never inline due to thread_local bugs
#[inline(never)]
fn data_init() -> *const Data {
    let data = hide(&DATA as *const Data);

    {
        let data = unsafe { &*data };
        if unlikely(!data.registered.get()) {
            data.register();
        }
    }

    data
}

// TODO: Can this be made into an OwnedPin type which restores the state when dropped?
// Would need to check that only one of them is around?
// Can we keep `pin` in that case? Could drop `OwnedPin` inside `pin`?
// Using seperate flags for both of them might work.
// Won't be optimized away like `pin`
// Use a reference count?

/// Marks the current thread as pinned and returns a proof of that to the closure.
///
/// This adds the current thread to the set of threads that needs to regularly call [collect]
/// before memory can be freed. [release] can be called if a thread no longer needs
/// access to lock-free data structures for an extended period of time.
#[inline]
pub fn pin<R>(f: impl FnOnce(Pin<'_>) -> R) -> R {
    let data = unsafe { &*(hide(data_init())) };
    let old_pinned = data.pinned.get();
    data.pinned.set(true);
    guard(old_pinned, |pin| data.pinned.set(*pin));
    f(Pin {
        _private: PhantomData,
    })
}

/// Removes the current thread from the threads allowed to access lock-free data structures.
///
/// This allows memory to be freed without waiting for [collect] calls from the current thread.
/// [pin] can be called to after to continue accessing lock-free data structures.
///
/// This will not free any garbage so [collect] should be called first before the last thread
/// terminates to avoid memory leaks.
///
/// This will panic if called while the current thread is pinned.
pub fn release() {
    let data = unsafe { &*(hide(data())) };
    if cfg!(debug_assertions) {
        data.assert_unpinned();
    }
    if data.registered.get() {
        data.assert_unpinned();
        data.registered.set(false);
        COLLECTOR.lock().unregister();
    }
}

/// Signals a quiescent state where garbage may be collected.
///
/// This may collect garbage using the callbacks registered in [Pin::defer_unchecked](struct.Pin.html#method.defer_unchecked).
///
/// This will panic if called while a thread is pinned.
pub fn collect() {
    let data = unsafe { &*(hide(data())) };
    if cfg!(debug_assertions) {
        data.assert_unpinned();
    }
    let new = EVENTS.load(Ordering::Acquire);
    if unlikely(new != data.seen_events.get()) {
        data.seen_events.set(new);
        cold_path(|| {
            data.assert_unpinned();

            let callbacks = {
                let mut collector = COLLECTOR.lock();

                // Check if we could block any deferred methods
                if data.registered.get() {
                    collector.quiet()
                } else {
                    collector.collect_unregistered()
                }
            };

            callbacks.into_iter().for_each(|callback| callback());
        });
    }
}

static COLLECTOR: LazyLock<Mutex<Collector>> = LazyLock::new(|| Mutex::new(Collector::new()));

type Callbacks = Vec<Box<dyn FnOnce() + Send>>;

struct Collector {
    pending: bool,
    busy_count: usize,
    threads: HashMap<ThreadId, bool>,
    current_deferred: Callbacks,
    previous_deferred: Callbacks,
}

impl Collector {
    fn new() -> Self {
        Self {
            pending: false,
            busy_count: 0,
            threads: HashMap::new(),
            current_deferred: Vec::new(),
            previous_deferred: Vec::new(),
        }
    }

    fn register(&mut self) {
        debug_assert!(!self.threads.contains_key(&thread::current().id()));

        self.busy_count += 1;
        self.threads.insert(thread::current().id(), false);
    }

    fn unregister(&mut self) {
        debug_assert!(self.threads.contains_key(&thread::current().id()));

        let thread = &thread::current().id();
        if *self.threads.get(&thread).unwrap() {
            self.busy_count -= 1;

            if self.busy_count == 0
                && (!self.previous_deferred.is_empty() || !self.current_deferred.is_empty())
            {
                // Don't collect garbage here, but let the other threads know that there's
                // garbage to be collected.
                self.pending = true;
                EVENTS.fetch_add(1, Ordering::Release);
            }
        }

        self.threads.remove(&thread);
    }

    fn collect_unregistered(&mut self) -> Callbacks {
        debug_assert!(!self.threads.contains_key(&thread::current().id()));

        if self.threads.is_empty() {
            let mut callbacks = mem::take(&mut self.previous_deferred);
            callbacks.extend(mem::take(&mut self.current_deferred));
            callbacks
        } else {
            Vec::new()
        }
    }

    fn quiet(&mut self) -> Callbacks {
        let quiet = self.threads.get_mut(&thread::current().id()).unwrap();
        if !*quiet || self.pending {
            if !*quiet {
                self.busy_count -= 1;
                *quiet = true;
            }

            if self.busy_count == 0 {
                // All threads are quiet
                self.pending = false;

                self.busy_count = self.threads.len();
                self.threads.values_mut().for_each(|value| {
                    *value = false;
                });

                let mut callbacks = mem::take(&mut self.previous_deferred);
                self.previous_deferred = mem::take(&mut self.current_deferred);

                if !self.previous_deferred.is_empty() {
                    // Mark ourselves as quiet again
                    callbacks.extend(self.quiet());

                    // Signal other threads to check in
                    EVENTS.fetch_add(1, Ordering::Release);
                }

                callbacks
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }

    fn defer(&mut self, callback: Box<dyn FnOnce() + Send>) {
        self.current_deferred.push(callback);
    }
}
