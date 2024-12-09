//! A `LazyKey` implementation using racy initialization.
//!
//! Unfortunately, none of the platforms currently supported by `std` allows
//! creating TLS keys at compile-time. Thus we need a way to lazily create keys.
//! Instead of blocking API like `OnceLock`, we use racy initialization, which
//! should be more lightweight and avoids circular dependencies with the rest of
//! `std`.

use crate::sync::atomic::{self, AtomicUsize, Ordering};

/// A type for TLS keys that are statically allocated.
///
/// This is basically a `LazyLock<Key>`, but avoids blocking and circular
/// dependencies with the rest of `std`.
pub struct LazyKey {
    /// Inner static TLS key (internals).
    key: AtomicUsize,
    /// Destructor for the TLS value.
    dtor: Option<unsafe extern "C" fn(*mut u8)>,
    /// Next element of process-wide destructor list.
    #[cfg(not(target_thread_local))]
    next: atomic::AtomicPtr<LazyKey>,
}

// Define a sentinel value that is likely not to be returned
// as a TLS key.
#[cfg(not(target_os = "nto"))]
const KEY_SENTVAL: usize = 0;
// On QNX Neutrino, 0 is always returned when currently not in use.
// Using 0 would mean to always create two keys and remote the first
// one (with value of 0) immediately afterwards.
#[cfg(target_os = "nto")]
const KEY_SENTVAL: usize = libc::PTHREAD_KEYS_MAX + 1;

impl LazyKey {
    pub const fn new(dtor: Option<unsafe extern "C" fn(*mut u8)>) -> LazyKey {
        LazyKey {
            key: atomic::AtomicUsize::new(KEY_SENTVAL),
            dtor,
            #[cfg(not(target_thread_local))]
            next: atomic::AtomicPtr::new(crate::ptr::null_mut()),
        }
    }

    #[inline]
    pub fn force(&'static self) -> super::Key {
        match self.key.load(Ordering::Acquire) {
            KEY_SENTVAL => self.lazy_init() as super::Key,
            n => n as super::Key,
        }
    }

    fn lazy_init(&'static self) -> usize {
        // POSIX allows the key created here to be KEY_SENTVAL, but the compare_exchange
        // below relies on using KEY_SENTVAL as a sentinel value to check who won the
        // race to set the shared TLS key. As far as I know, there is no
        // guaranteed value that cannot be returned as a posix_key_create key,
        // so there is no value we can initialize the inner key with to
        // prove that it has not yet been set. As such, we'll continue using a
        // value of KEY_SENTVAL, but with some gyrations to make sure we have a non-KEY_SENTVAL
        // value returned from the creation routine.
        // FIXME: this is clearly a hack, and should be cleaned up.
        let key1 = super::create(self.dtor);
        let key = if key1 as usize != KEY_SENTVAL {
            key1
        } else {
            let key2 = super::create(self.dtor);
            unsafe {
                super::destroy(key1);
            }
            key2
        };
        rtassert!(key as usize != KEY_SENTVAL);
        match self.key.compare_exchange(
            KEY_SENTVAL,
            key as usize,
            Ordering::Release,
            Ordering::Acquire,
        ) {
            // The CAS succeeded, so we've created the actual key
            Ok(_) => {
                #[cfg(not(target_thread_local))]
                if self.dtor.is_some() {
                    unsafe { register_dtor(self) };
                }
                key as usize
            }
            // If someone beat us to the punch, use their key instead
            Err(n) => unsafe {
                super::destroy(key);
                n
            },
        }
    }
}

/// POSIX does not run TLS destructors on process exit.
/// Thus we keep our own global list for that purpose.
#[cfg(not(target_thread_local))]
static DTORS: atomic::AtomicPtr<LazyKey> = atomic::AtomicPtr::new(crate::ptr::null_mut());

/// Registers destructor to run at process exit.
#[cfg(not(target_thread_local))]
unsafe fn register_dtor(key: &'static LazyKey) {
    crate::sys::thread_local::guard::enable();

    let this = <*const LazyKey>::cast_mut(key);
    // Use acquire ordering to pass along the changes done by the previously
    // registered keys when we store the new head with release ordering.
    let mut head = DTORS.load(Ordering::Acquire);
    loop {
        key.next.store(head, Ordering::Relaxed);
        match DTORS.compare_exchange_weak(head, this, Ordering::Release, Ordering::Acquire) {
            Ok(_) => break,
            Err(new) => head = new,
        }
    }
}

/// Run destructors at process exit.
///
/// SAFETY: This will and must only be run by the destructor callback in [`guard`].
#[cfg(not(target_thread_local))]
pub unsafe fn run_dtors() {
    for _ in 0..5 {
        let mut any_run = false;

        // Use acquire ordering to observe key initialization.
        let mut cur = DTORS.load(Ordering::Acquire);
        while !cur.is_null() {
            let key = unsafe { (*cur).key.load(Ordering::Acquire) };
            let dtor = unsafe { (*cur).dtor.unwrap() };
            cur = unsafe { (*cur).next.load(Ordering::Relaxed) };

            let ptr = unsafe { super::get(key as _) };
            if !ptr.is_null() {
                unsafe {
                    super::set(key as _, crate::ptr::null_mut());
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
