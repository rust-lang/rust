//! A TLS destructor system.
//!
//! Turns out, like pretty much everything, Windows is pretty close the
//! functionality that Unix provides, but slightly different! In the case of
//! TLS, Windows does not provide an API to provide a destructor for a TLS
//! variable. This ends up being pretty crucial to this implementation, so we
//! need a way around this.
//!
//! The solution here ended up being a little obscure, but fear not, the
//! internet has informed me [1][2] that this solution is not unique (no way
//! I could have thought of it as well!). The key idea is to insert some hook
//! somewhere to run arbitrary code on thread termination. With this in place
//! we'll be able to run anything we like, including all TLS destructors!
//!
//! If you're looking at this code, and wondering "what is this doing?",
//! you're not alone! I'll try to break this down step by step:
//!
//! # What's up with CRT$XLB?
//!
//! For anything about TLS destructors to work on Windows, we have to be able
//! to run *something* when a thread exits. To do so, we place a very special
//! static in a very special location. If this is encoded in just the right
//! way, the kernel's loader is apparently nice enough to run some function
//! of ours whenever a thread exits! How nice of the kernel!
//!
//! Lots of detailed information can be found in source [1] above, but the
//! gist of it is that this is leveraging a feature of Microsoft's PE format
//! (executable format) which is not actually used by any compilers today.
//! This apparently translates to any callbacks in the ".CRT$XLB" section
//! being run on certain events.
//!
//! So after all that, we use the compiler's #[link_section] feature to place
//! a callback pointer into the magic section so it ends up being called.
//!
//! # What's up with this callback?
//!
//! The callback specified receives a number of parameters from... someone!
//! (the kernel? the runtime? I'm not quite sure!) There are a few events that
//! this gets invoked for, but we're currently only interested on when a
//! thread or a process "detaches" (exits). The process part happens for the
//! last thread and the thread part happens for any normal thread.
//!
//! # The article mentions weird stuff about "/INCLUDE"?
//!
//! It sure does! Specifically we're talking about this quote:
//!
//! > The Microsoft run-time library facilitates this process by defining a
//! > memory image of the TLS Directory and giving it the special name
//! > “__tls_used” (Intel x86 platforms) or “_tls_used” (other platforms). The
//! > linker looks for this memory image and uses the data there to create the
//! > TLS Directory. Other compilers that support TLS and work with the
//! > Microsoft linker must use this same technique.
//!
//! Basically what this means is that if we want support for our TLS
//! destructors/our hook being called then we need to make sure the linker does
//! not omit this symbol. Otherwise it will omit it and our callback won't be
//! wired up.
//!
//! We don't actually use the `/INCLUDE` linker flag here like the article
//! mentions because the Rust compiler doesn't propagate linker flags, but
//! instead we use a shim function which performs a volatile 1-byte load from
//! the address of the symbol to ensure it sticks around.
//!
//! [1]: https://www.codeproject.com/Articles/8113/Thread-Local-Storage-The-C-Way
//! [2]: https://github.com/ChromiumWebApps/chromium/blob/master/base/threading/thread_local_storage_win.cc#L42

#![unstable(feature = "thread_local_internals", issue = "none")]

use crate::ptr;
use crate::sync::atomic::{
    AtomicBool,
    Ordering::{Acquire, Relaxed},
};
use crate::sys::c;

// If the target uses native TLS, run its destructors.
#[cfg(target_thread_local)]
use crate::sys::common::thread_local::run_dtors;
// Otherwise, run the destructors for the key-based variant.
#[cfg(not(target_thread_local))]
use super::thread_local_key::run_dtors;

/// An optimization hint. The compiler is often smart enough to know if an atomic
/// is never set and can remove dead code based on that fact.
static HAS_DTORS: AtomicBool = AtomicBool::new(false);

/// Ensure that thread-locals are destroyed when the thread exits.
pub fn activate() {
    HAS_DTORS.store(true, Relaxed);
}

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
        run_dtors(ptr::null_mut());
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
