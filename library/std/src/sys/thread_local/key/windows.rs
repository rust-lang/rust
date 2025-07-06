//! Implementation of `LazyKey` for Windows.
//!
//! Windows has no native support for running destructors so we manage our own
//! list of destructors to keep track of how to destroy keys. We then install a
//! callback later to get invoked whenever a thread exits, running all
//! appropriate destructors (see the [`guard`](guard) module documentation).
//!
//! This will likely need to be improved over time, but this module attempts a
//! "poor man's" destructor callback system. Once we've got a list of what to
//! run, we iterate over all keys, check their values, and then run destructors
//! if the values turn out to be non null (setting them to null just beforehand).
//! We do this a few times in a loop to basically match Unix semantics. If we
//! don't reach a fixed point after a short while then we just inevitably leak
//! something.
//!
//! The list is implemented as an atomic single-linked list of `LazyKey`s and
//! does not support unregistration. Unfortunately, this means that we cannot
//! use racy initialization for creating the keys in `LazyKey`, as that could
//! result in destructors being missed. Hence, we synchronize the creation of
//! keys with destructors through [`INIT_ONCE`](c::INIT_ONCE) (`std`'s
//! [`Once`](crate::sync::Once) cannot be used since it might use TLS itself).
//! For keys without destructors, racy initialization suffices.

// FIXME: investigate using a fixed-size array instead, as the maximum number
//        of keys is [limited to 1088](https://learn.microsoft.com/en-us/windows/win32/ProcThread/thread-local-storage).

use crate::cell::UnsafeCell;
use crate::ptr;
use crate::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
use crate::sync::atomic::{Atomic, AtomicPtr, AtomicU32};
use crate::sys::c;
use crate::sys::thread_local::guard;

pub type Key = u32;
type Dtor = unsafe extern "C" fn(*mut u8);

pub struct LazyKey {
    /// The key value shifted up by one. Since TLS_OUT_OF_INDEXES == u32::MAX
    /// is not a valid key value, this allows us to use zero as sentinel value
    /// without risking overflow.
    key: Atomic<Key>,
    dtor: Option<Dtor>,
    next: Atomic<*mut LazyKey>,
    /// Currently, destructors cannot be unregistered, so we cannot use racy
    /// initialization for keys. Instead, we need synchronize initialization.
    /// Use the Windows-provided `Once` since it does not require TLS.
    once: UnsafeCell<c::INIT_ONCE>,
}

impl LazyKey {
    #[inline]
    pub const fn new(dtor: Option<Dtor>) -> LazyKey {
        LazyKey {
            key: AtomicU32::new(0),
            dtor,
            next: AtomicPtr::new(ptr::null_mut()),
            once: UnsafeCell::new(c::INIT_ONCE_STATIC_INIT),
        }
    }

    #[inline]
    pub fn force(&'static self) -> Key {
        match self.key.load(Acquire) {
            0 => unsafe { self.init() },
            key => key - 1,
        }
    }

    #[cold]
    unsafe fn init(&'static self) -> Key {
        if self.dtor.is_some() {
            let mut pending = c::FALSE;
            let r = unsafe {
                c::InitOnceBeginInitialize(self.once.get(), 0, &mut pending, ptr::null_mut())
            };
            assert_eq!(r, c::TRUE);

            if pending == c::FALSE {
                // Some other thread initialized the key, load it.
                self.key.load(Relaxed) - 1
            } else {
                let key = unsafe { c::TlsAlloc() };
                if key == c::TLS_OUT_OF_INDEXES {
                    // Since we abort the process, there is no need to wake up
                    // the waiting threads. If this were a panic, the wakeup
                    // would need to occur first in order to avoid deadlock.
                    rtabort!("out of TLS indexes");
                }

                unsafe {
                    register_dtor(self);
                }

                // Release-storing the key needs to be the last thing we do.
                // This is because in `fn key()`, other threads will do an acquire load of the key,
                // and if that sees this write then it will entirely bypass the `InitOnce`. We thus
                // need to establish synchronization through `key`. In particular that acquire load
                // must happen-after the register_dtor above, to ensure the dtor actually runs!
                self.key.store(key + 1, Release);

                let r = unsafe { c::InitOnceComplete(self.once.get(), 0, ptr::null_mut()) };
                debug_assert_eq!(r, c::TRUE);

                key
            }
        } else {
            // If there is no destructor to clean up, we can use racy initialization.

            let key = unsafe { c::TlsAlloc() };
            if key == c::TLS_OUT_OF_INDEXES {
                rtabort!("out of TLS indexes");
            }

            match self.key.compare_exchange(0, key + 1, AcqRel, Acquire) {
                Ok(_) => key,
                Err(new) => unsafe {
                    // Some other thread completed initialization first, so destroy
                    // our key and use theirs.
                    let r = c::TlsFree(key);
                    debug_assert_eq!(r, c::TRUE);
                    new - 1
                },
            }
        }
    }
}

unsafe impl Send for LazyKey {}
unsafe impl Sync for LazyKey {}

#[inline]
pub unsafe fn set(key: Key, val: *mut u8) {
    let r = unsafe { c::TlsSetValue(key, val.cast()) };
    debug_assert_eq!(r, c::TRUE);
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    unsafe { c::TlsGetValue(key).cast() }
}

static DTORS: Atomic<*mut LazyKey> = AtomicPtr::new(ptr::null_mut());

/// Should only be called once per key, otherwise loops or breaks may occur in
/// the linked list.
unsafe fn register_dtor(key: &'static LazyKey) {
    guard::enable();

    let this = <*const LazyKey>::cast_mut(key);
    // Use acquire ordering to pass along the changes done by the previously
    // registered keys when we store the new head with release ordering.
    let mut head = DTORS.load(Acquire);
    loop {
        key.next.store(head, Relaxed);
        match DTORS.compare_exchange_weak(head, this, Release, Acquire) {
            Ok(_) => break,
            Err(new) => head = new,
        }
    }
}

/// This will and must only be run by the destructor callback in [`guard`].
pub unsafe fn run_dtors() {
    for _ in 0..5 {
        let mut any_run = false;

        // Use acquire ordering to observe key initialization.
        let mut cur = DTORS.load(Acquire);
        while !cur.is_null() {
            let pre_key = unsafe { (*cur).key.load(Acquire) };
            let dtor = unsafe { (*cur).dtor.unwrap() };
            cur = unsafe { (*cur).next.load(Relaxed) };

            // In LazyKey::init, we register the dtor before setting `key`.
            // So if one thread's `run_dtors` races with another thread executing `init` on the same
            // `LazyKey`, we can encounter a key of 0 here. That means this key was never
            // initialized in this thread so we can safely skip it.
            if pre_key == 0 {
                continue;
            }
            // If this is non-zero, then via the `Acquire` load above we synchronized with
            // everything relevant for this key. (It's not clear that this is needed, since the
            // release-acquire pair on DTORS also establishes synchronization, but better safe than
            // sorry.)
            let key = pre_key - 1;

            let ptr = unsafe { c::TlsGetValue(key) };
            if !ptr.is_null() {
                unsafe {
                    c::TlsSetValue(key, ptr::null_mut());
                    dtor(ptr as *mut _);
                    any_run = true;
                }
            }
        }

        if !any_run {
            break;
        }
    }
}
