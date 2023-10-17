#![cfg(not(target_thread_local))]

use crate::cell::UnsafeCell;
use crate::ptr;
use crate::sync::atomic::{
    AtomicPtr, AtomicU32,
    Ordering::{AcqRel, Acquire, Relaxed, Release},
};
use crate::sys::c;

#[cfg(test)]
mod tests;

type Key = c::DWORD;
type Dtor = unsafe extern "C" fn(*mut u8);

pub struct StaticKey {
    /// The key value shifted up by one. Since TLS_OUT_OF_INDEXES == DWORD::MAX
    /// is not a valid key value, this allows us to use zero as sentinel value
    /// without risking overflow.
    key: AtomicU32,
    dtor: Option<Dtor>,
    next: AtomicPtr<StaticKey>,
    /// Currently, destructors cannot be unregistered, so we cannot use racy
    /// initialization for keys. Instead, we need synchronize initialization.
    /// Use the Windows-provided `Once` since it does not require TLS.
    once: UnsafeCell<c::INIT_ONCE>,
}

impl StaticKey {
    #[inline]
    pub const fn new(dtor: Option<Dtor>) -> StaticKey {
        StaticKey {
            key: AtomicU32::new(0),
            dtor,
            next: AtomicPtr::new(ptr::null_mut()),
            once: UnsafeCell::new(c::INIT_ONCE_STATIC_INIT),
        }
    }

    #[inline]
    pub unsafe fn set(&'static self, val: *mut u8) {
        let r = c::TlsSetValue(self.key(), val.cast());
        debug_assert_eq!(r, c::TRUE);
    }

    #[inline]
    pub unsafe fn get(&'static self) -> *mut u8 {
        c::TlsGetValue(self.key()).cast()
    }

    #[inline]
    unsafe fn key(&'static self) -> Key {
        match self.key.load(Acquire) {
            0 => self.init(),
            key => key - 1,
        }
    }

    #[cold]
    unsafe fn init(&'static self) -> Key {
        if self.dtor.is_some() {
            let mut pending = c::FALSE;
            let r = c::InitOnceBeginInitialize(self.once.get(), 0, &mut pending, ptr::null_mut());
            assert_eq!(r, c::TRUE);

            if pending == c::FALSE {
                // Some other thread initialized the key, load it.
                self.key.load(Relaxed) - 1
            } else {
                let key = c::TlsAlloc();
                if key == c::TLS_OUT_OF_INDEXES {
                    // Wakeup the waiting threads before panicking to avoid deadlock.
                    c::InitOnceComplete(self.once.get(), c::INIT_ONCE_INIT_FAILED, ptr::null_mut());
                    panic!("out of TLS indexes");
                }

                self.key.store(key + 1, Release);
                register_dtor(self);

                let r = c::InitOnceComplete(self.once.get(), 0, ptr::null_mut());
                debug_assert_eq!(r, c::TRUE);

                key
            }
        } else {
            // If there is no destructor to clean up, we can use racy initialization.

            let key = c::TlsAlloc();
            assert_ne!(key, c::TLS_OUT_OF_INDEXES, "out of TLS indexes");

            match self.key.compare_exchange(0, key + 1, AcqRel, Acquire) {
                Ok(_) => key,
                Err(new) => {
                    // Some other thread completed initialization first, so destroy
                    // our key and use theirs.
                    let r = c::TlsFree(key);
                    debug_assert_eq!(r, c::TRUE);
                    new - 1
                }
            }
        }
    }
}

unsafe impl Send for StaticKey {}
unsafe impl Sync for StaticKey {}

// -------------------------------------------------------------------------
// Dtor registration
//
// Windows has no native support for running destructors so we manage our own
// list of destructors to keep track of how to destroy keys. We then install a
// callback later to get invoked whenever a thread exits, running all
// appropriate destructors.
//
// Currently unregistration from this list is not supported. A destructor can be
// registered but cannot be unregistered. There's various simplifying reasons
// for doing this, the big ones being:
//
// 1. Currently we don't even support deallocating TLS keys, so normal operation
//    doesn't need to deallocate a destructor.
// 2. There is no point in time where we know we can unregister a destructor
//    because it could always be getting run by some remote thread.
//
// Typically processes have a statically known set of TLS keys which is pretty
// small, and we'd want to keep this memory alive for the whole process anyway
// really.

static DTORS: AtomicPtr<StaticKey> = AtomicPtr::new(ptr::null_mut());

/// Should only be called once per key, otherwise loops or breaks may occur in
/// the linked list.
unsafe fn register_dtor(key: &'static StaticKey) {
    // Ensure this is never run when native thread locals are available.
    assert_eq!(false, cfg!(target_thread_local));
    let this = <*const StaticKey>::cast_mut(key);
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
    super::thread_local_guard::activate();
}

// What's up with running all these destructors?
//
// This will likely need to be improved over time, but this function
// attempts a "poor man's" destructor callback system. Once we've got a list
// of what to run, we iterate over all keys, check their values, and then run
// destructors if the values turn out to be non null (setting them to null just
// beforehand). We do this a few times in a loop to basically match Unix
// semantics. If we don't reach a fixed point after a short while then we just
// inevitably leak something most likely.
pub(super) unsafe fn run_dtors(_ptr: *mut u8) {
    for _ in 0..5 {
        let mut any_run = false;

        // Use acquire ordering to observe key initialization.
        let mut cur = DTORS.load(Acquire);
        while !cur.is_null() {
            let key = (*cur).key.load(Relaxed) - 1;
            let dtor = (*cur).dtor.unwrap();

            let ptr = c::TlsGetValue(key);
            if !ptr.is_null() {
                c::TlsSetValue(key, ptr::null_mut());
                dtor(ptr as *mut _);
                any_run = true;
            }

            cur = (*cur).next.load(Relaxed);
        }

        if !any_run {
            break;
        }
    }
}
