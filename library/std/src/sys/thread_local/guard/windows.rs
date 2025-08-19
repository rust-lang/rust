//! Support for Windows TLS destructors.
//!
//! Unfortunately, Windows does not provide a nice API to provide a destructor
//! for a TLS variable. Thus, the solution here ended up being a little more
//! obscure, but fear not, the internet has informed me [1][2] that this solution
//! is not unique (no way I could have thought of it as well!). The key idea is
//! to insert some hook somewhere to run arbitrary code on thread termination.
//! With this in place we'll be able to run anything we like, including all
//! TLS destructors!
//!
//! In order to realize this, all TLS destructors are tracked by *us*, not the
//! Windows runtime. This means that we have a global list of destructors for
//! each TLS key or variable that we know about.
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
//! So after all that, we use the compiler's `#[link_section]` feature to place
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
//! ```quote
//! The Microsoft run-time library facilitates this process by defining a
//! memory image of the TLS Directory and giving it the special name
//! “__tls_used” (Intel x86 platforms) or “_tls_used” (other platforms). The
//! linker looks for this memory image and uses the data there to create the
//! TLS Directory. Other compilers that support TLS and work with the
//! Microsoft linker must use this same technique.
//! ```
//!
//! Basically what this means is that if we want support for our TLS
//! destructors/our hook being called then we need to make sure the linker does
//! not omit this symbol. Otherwise it will omit it and our callback won't be
//! wired up.
//!
//! We don't actually use the `/INCLUDE` linker flag here like the article
//! mentions because the Rust compiler doesn't propagate linker flags, but
//! instead we use a shim function which performs a volatile 1-byte load from
//! the address of the _tls_used symbol to ensure it sticks around.
//!
//! [1]: https://www.codeproject.com/Articles/8113/Thread-Local-Storage-The-C-Way
//! [2]: https://github.com/ChromiumWebApps/chromium/blob/master/base/threading/thread_local_storage_win.cc#L42

use core::ffi::c_void;

use crate::ptr;
use crate::sys::c;

unsafe extern "C" {
    #[link_name = "_tls_used"]
    static TLS_USED: u8;
}
pub fn enable() {
    // When destructors are used, we need to add a reference to the _tls_used
    // symbol provided by the CRT, otherwise the TLS support code will get
    // GC'd by the linker and our callback won't be called.
    unsafe { ptr::from_ref(&TLS_USED).read_volatile() };
    // We also need to reference CALLBACK to make sure it does not get GC'd
    // by the compiler/LLVM. The callback will end up inside the TLS
    // callback array pointed to by _TLS_USED through linker shenanigans,
    // but as far as the compiler is concerned, it looks like the data is
    // unused, so we need this hack to prevent it from disappearing.
    unsafe { ptr::from_ref(&CALLBACK).read_volatile() };
}

#[unsafe(link_section = ".CRT$XLB")]
#[cfg_attr(miri, used)] // Miri only considers explicitly `#[used]` statics for `lookup_link_section`
pub static CALLBACK: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) = tls_callback;

unsafe extern "system" fn tls_callback(_h: *mut c_void, dw_reason: u32, _pv: *mut c_void) {
    if dw_reason == c::DLL_THREAD_DETACH || dw_reason == c::DLL_PROCESS_DETACH {
        unsafe {
            #[cfg(target_thread_local)]
            super::super::destructors::run();
            #[cfg(not(target_thread_local))]
            super::super::key::run_dtors();

            crate::rt::thread_cleanup();
        }
    }
}
