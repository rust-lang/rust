use crate::cell::UnsafeCell;
use crate::ptr;
use crate::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicU32,
    Ordering::{AcqRel, Acquire, Relaxed, Release},
};
use crate::sys::c;

#[cfg(test)]
mod tests;

/// An optimization hint. The compiler is often smart enough to know if an atomic
/// is never set and can remove dead code based on that fact.
static HAS_DTORS: AtomicBool = AtomicBool::new(false);

// Using a per-thread list avoids the problems in synchronizing global state.
#[thread_local]
#[cfg(target_thread_local)]
static mut DESTRUCTORS: Vec<(*mut u8, unsafe extern "C" fn(*mut u8))> = Vec::new();

// Ensure this can never be inlined because otherwise this may break in dylibs.
// See #44391.
#[inline(never)]
#[cfg(target_thread_local)]
pub unsafe fn register_keyless_dtor(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    DESTRUCTORS.push((t, dtor));
    HAS_DTORS.store(true, Relaxed);
}

#[inline(never)] // See comment above
#[cfg(target_thread_local)]
/// Runs destructors. This should not be called until thread exit.
unsafe fn run_keyless_dtors() {
    // Drop all the destructors.
    //
    // Note: While this is potentially an infinite loop, it *should* be
    // the case that this loop always terminates because we provide the
    // guarantee that a TLS key cannot be set after it is flagged for
    // destruction.
    while let Some((ptr, dtor)) = DESTRUCTORS.pop() {
        (dtor)(ptr);
    }
    // We're done so free the memory.
    DESTRUCTORS = Vec::new();
}

type Key = c::DWORD;
type Dtor = unsafe extern "C" fn(*mut u8);

// Turns out, like pretty much everything, Windows is pretty close the
// functionality that Unix provides, but slightly different! In the case of
// TLS, Windows does not provide an API to provide a destructor for a TLS
// variable. This ends up being pretty crucial to this implementation, so we
// need a way around this.
//
// The solution here ended up being a little obscure, but fear not, the
// internet has informed me [1][2] that this solution is not unique (no way
// I could have thought of it as well!). The key idea is to insert some hook
// somewhere to run arbitrary code on thread termination. With this in place
// we'll be able to run anything we like, including all TLS destructors!
//
// To accomplish this feat, we perform a number of threads, all contained
// within this module:
//
// * All TLS destructors are tracked by *us*, not the Windows runtime. This
//   means that we have a global list of destructors for each TLS key that
//   we know about.
// * When a thread exits, we run over the entire list and run dtors for all
//   non-null keys. This attempts to match Unix semantics in this regard.
//
// For more details and nitty-gritty, see the code sections below!
//
// [1]: https://www.codeproject.com/Articles/8113/Thread-Local-Storage-The-C-Way
// [2]: https://github.com/ChromiumWebApps/chromium/blob/master/base/threading/thread_local_storage_win.cc#L42

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
    HAS_DTORS.store(true, Release);
}

// -------------------------------------------------------------------------
// Where the Magic (TM) Happens
//
// If you're looking at this code, and wondering "what is this doing?",
// you're not alone! I'll try to break this down step by step:
//
// # What's up with CRT$XLB?
//
// For anything about TLS destructors to work on Windows, we have to be able
// to run *something* when a thread exits. To do so, we place a very special
// static in a very special location. If this is encoded in just the right
// way, the kernel's loader is apparently nice enough to run some function
// of ours whenever a thread exits! How nice of the kernel!
//
// Lots of detailed information can be found in source [1] above, but the
// gist of it is that this is leveraging a feature of Microsoft's PE format
// (executable format) which is not actually used by any compilers today.
// This apparently translates to any callbacks in the ".CRT$XLB" section
// being run on certain events.
//
// So after all that, we use the compiler's #[link_section] feature to place
// a callback pointer into the magic section so it ends up being called.
//
// # What's up with this callback?
//
// The callback specified receives a number of parameters from... someone!
// (the kernel? the runtime? I'm not quite sure!) There are a few events that
// this gets invoked for, but we're currently only interested on when a
// thread or a process "detaches" (exits). The process part happens for the
// last thread and the thread part happens for any normal thread.
//
// # Ok, what's up with running all these destructors?
//
// This will likely need to be improved over time, but this function
// attempts a "poor man's" destructor callback system. Once we've got a list
// of what to run, we iterate over all keys, check their values, and then run
// destructors if the values turn out to be non null (setting them to null just
// beforehand). We do this a few times in a loop to basically match Unix
// semantics. If we don't reach a fixed point after a short while then we just
// inevitably leak something most likely.
//
// # The article mentions weird stuff about "/INCLUDE"?
//
// It sure does! Specifically we're talking about this quote:
//
//      The Microsoft run-time library facilitates this process by defining a
//      memory image of the TLS Directory and giving it the special name
//      “__tls_used” (Intel x86 platforms) or “_tls_used” (other platforms). The
//      linker looks for this memory image and uses the data there to create the
//      TLS Directory. Other compilers that support TLS and work with the
//      Microsoft linker must use this same technique.
//
// Basically what this means is that if we want support for our TLS
// destructors/our hook being called then we need to make sure the linker does
// not omit this symbol. Otherwise it will omit it and our callback won't be
// wired up.
//
// We don't actually use the `/INCLUDE` linker flag here like the article
// mentions because the Rust compiler doesn't propagate linker flags, but
// instead we use a shim function which performs a volatile 1-byte load from
// the address of the symbol to ensure it sticks around.

#[link_section = ".CRT$XLB"]
#[allow(dead_code, unused_variables)]
#[used] // we don't want LLVM eliminating this symbol for any reason, and
// when the symbol makes it to the linker the linker will take over
pub static p_thread_callback: unsafe extern "system" fn(c::LPVOID, c::DWORD, c::LPVOID) =
    on_tls_callback;

#[allow(dead_code, unused_variables)]
unsafe extern "system" fn on_tls_callback(h: c::LPVOID, dwReason: c::DWORD, pv: c::LPVOID) {
    if !HAS_DTORS.load(Acquire) {
        return;
    }
    if dwReason == c::DLL_THREAD_DETACH || dwReason == c::DLL_PROCESS_DETACH {
        #[cfg(not(target_thread_local))]
        run_dtors();
        #[cfg(target_thread_local)]
        run_keyless_dtors();
    }

    // See comments above for what this is doing. Note that we don't need this
    // trickery on GNU windows, just on MSVC.
    reference_tls_used();
    #[cfg(target_env = "msvc")]
    unsafe fn reference_tls_used() {
        extern "C" {
            static _tls_used: u8;
        }
        crate::intrinsics::volatile_load(&_tls_used);
    }
    #[cfg(not(target_env = "msvc"))]
    unsafe fn reference_tls_used() {}
}

#[allow(dead_code)] // actually called below
unsafe fn run_dtors() {
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
