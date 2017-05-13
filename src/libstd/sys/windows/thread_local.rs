// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ptr;
use sys::c;
use sys_common::mutex::Mutex;
use sys_common;

pub type Key = c::DWORD;
pub type Dtor = unsafe extern fn(*mut u8);

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
// * All TLS destructors are tracked by *us*, not the windows runtime. This
//   means that we have a global list of destructors for each TLS key that
//   we know about.
// * When a TLS key is destroyed, we're sure to remove it from the dtor list
//   if it's in there.
// * When a thread exits, we run over the entire list and run dtors for all
//   non-null keys. This attempts to match Unix semantics in this regard.
//
// This ends up having the overhead of using a global list, having some
// locks here and there, and in general just adding some more code bloat. We
// attempt to optimize runtime by forgetting keys that don't have
// destructors, but this only gets us so far.
//
// For more details and nitty-gritty, see the code sections below!
//
// [1]: http://www.codeproject.com/Articles/8113/Thread-Local-Storage-The-C-Way
// [2]: https://github.com/ChromiumWebApps/chromium/blob/master/base
//                        /threading/thread_local_storage_win.cc#L42

// NB these are specifically not types from `std::sync` as they currently rely
// on poisoning and this module needs to operate at a lower level than requiring
// the thread infrastructure to be in place (useful on the borders of
// initialization/destruction).
static DTOR_LOCK: Mutex = Mutex::new();
static mut DTORS: *mut Vec<(Key, Dtor)> = ptr::null_mut();

// -------------------------------------------------------------------------
// Native bindings
//
// This section is just raw bindings to the native functions that Windows
// provides, There's a few extra calls to deal with destructors.

#[inline]
pub unsafe fn create(dtor: Option<Dtor>) -> Key {
    let key = c::TlsAlloc();
    assert!(key != c::TLS_OUT_OF_INDEXES);
    if let Some(f) = dtor {
        register_dtor(key, f);
    }
    return key;
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    let r = c::TlsSetValue(key, value as c::LPVOID);
    debug_assert!(r != 0);
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    c::TlsGetValue(key) as *mut u8
}

#[inline]
pub unsafe fn destroy(key: Key) {
    if unregister_dtor(key) {
        // FIXME: Currently if a key has a destructor associated with it we
        // can't actually ever unregister it. If we were to
        // unregister it, then any key destruction would have to be
        // serialized with respect to actually running destructors.
        //
        // We want to avoid a race where right before run_dtors runs
        // some destructors TlsFree is called. Allowing the call to
        // TlsFree would imply that the caller understands that *all
        // known threads* are not exiting, which is quite a difficult
        // thing to know!
        //
        // For now we just leak all keys with dtors to "fix" this.
        // Note that source [2] above shows precedent for this sort
        // of strategy.
    } else {
        let r = c::TlsFree(key);
        debug_assert!(r != 0);
    }
}

// -------------------------------------------------------------------------
// Dtor registration
//
// These functions are associated with registering and unregistering
// destructors. They're pretty simple, they just push onto a vector and scan
// a vector currently.
//
// FIXME: This could probably be at least a little faster with a BTree.

unsafe fn init_dtors() {
    if !DTORS.is_null() { return }

    let dtors = box Vec::<(Key, Dtor)>::new();

    let res = sys_common::at_exit(move|| {
        DTOR_LOCK.lock();
        let dtors = DTORS;
        DTORS = 1 as *mut _;
        Box::from_raw(dtors);
        assert!(DTORS as usize == 1); // can't re-init after destructing
        DTOR_LOCK.unlock();
    });
    if res.is_ok() {
        DTORS = Box::into_raw(dtors);
    } else {
        DTORS = 1 as *mut _;
    }
}

unsafe fn register_dtor(key: Key, dtor: Dtor) {
    DTOR_LOCK.lock();
    init_dtors();
    assert!(DTORS as usize != 0);
    assert!(DTORS as usize != 1,
            "cannot create new TLS keys after the main thread has exited");
    (*DTORS).push((key, dtor));
    DTOR_LOCK.unlock();
}

unsafe fn unregister_dtor(key: Key) -> bool {
    DTOR_LOCK.lock();
    init_dtors();
    assert!(DTORS as usize != 0);
    assert!(DTORS as usize != 1,
            "cannot unregister destructors after the main thread has exited");
    let ret = {
        let dtors = &mut *DTORS;
        let before = dtors.len();
        dtors.retain(|&(k, _)| k != key);
        dtors.len() != before
    };
    DTOR_LOCK.unlock();
    ret
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
// attempts a "poor man's" destructor callback system. To do this we clone a
// local copy of the dtor list to start out with. This is our fudgy attempt
// to not hold the lock while destructors run and not worry about the list
// changing while we're looking at it.
//
// Once we've got a list of what to run, we iterate over all keys, check
// their values, and then run destructors if the values turn out to be non
// null (setting them to null just beforehand). We do this a few times in a
// loop to basically match Unix semantics. If we don't reach a fixed point
// after a short while then we just inevitably leak something most likely.
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
#[linkage = "external"]
#[allow(dead_code, unused_variables)]
pub static p_thread_callback: unsafe extern "system" fn(c::LPVOID, c::DWORD,
                                                        c::LPVOID) =
        on_tls_callback;

#[allow(dead_code, unused_variables)]
unsafe extern "system" fn on_tls_callback(h: c::LPVOID,
                                          dwReason: c::DWORD,
                                          pv: c::LPVOID) {
    if dwReason == c::DLL_THREAD_DETACH || dwReason == c::DLL_PROCESS_DETACH {
        run_dtors();
    }

    // See comments above for what this is doing. Note that we don't need this
    // trickery on GNU windows, just on MSVC.
    reference_tls_used();
    #[cfg(target_env = "msvc")]
    unsafe fn reference_tls_used() {
        extern { static _tls_used: u8; }
        ::intrinsics::volatile_load(&_tls_used);
    }
    #[cfg(not(target_env = "msvc"))]
    unsafe fn reference_tls_used() {}
}

#[allow(dead_code)] // actually called above
unsafe fn run_dtors() {
    let mut any_run = true;
    for _ in 0..5 {
        if !any_run { break }
        any_run = false;
        let dtors = {
            DTOR_LOCK.lock();
            let ret = if DTORS as usize <= 1 {
                Vec::new()
            } else {
                (*DTORS).iter().map(|s| *s).collect()
            };
            DTOR_LOCK.unlock();
            ret
        };
        for &(key, dtor) in &dtors {
            let ptr = c::TlsGetValue(key);
            if !ptr.is_null() {
                c::TlsSetValue(key, ptr::null_mut());
                dtor(ptr as *mut _);
                any_run = true;
            }
        }
    }
}
