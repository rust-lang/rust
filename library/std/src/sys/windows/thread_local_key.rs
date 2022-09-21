use crate::mem;
use crate::ptr;
use crate::sync::atomic::{
    compiler_fence, AtomicPtr, AtomicUsize,
    Ordering::{Relaxed, Release},
};
use crate::sys::c;

pub type Key = c::DWORD;
pub type Dtor = unsafe extern "C" fn(*mut u8);

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
// Since the maximum number of keys is 1088 [3] and key values are always lower
// than 1088 [4], we can just use a static array to store the destructor functions
// and use the TLS key as index. This avoids all synchronization problems
// encountered with linked lists or other kinds of storage.
//
// For more details and nitty-gritty, see the code sections below!
//
// [1]: https://www.codeproject.com/Articles/8113/Thread-Local-Storage-The-C-Way
// [2]: https://github.com/ChromiumWebApps/chromium/blob/master/base/threading/thread_local_storage_win.cc#L42
// [3]: https://learn.microsoft.com/en-us/windows/win32/procthread/thread-local-storage
// [4]: https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-tlssetvalue

static DTORS: [AtomicPtr<()>; 1088] = [const { AtomicPtr::new(ptr::null_mut()) }; 1088];
// The highest key that has a destructor associated with it. Used as an
// optimization so we don't need to iterate over the whole array when there
// are only a few keys.
static HIGHEST: AtomicUsize = AtomicUsize::new(0);

#[inline]
pub unsafe fn create(dtor: Option<Dtor>) -> Key {
    let key = c::TlsAlloc();
    assert!(key != c::TLS_OUT_OF_INDEXES);

    if let Some(dtor) = dtor {
        DTORS[key as usize].store(mem::transmute::<Dtor, *mut ()>(dtor), Relaxed);
        HIGHEST.fetch_max(key as usize, Relaxed);
        // If the destructors are run in a signal handler running after this
        // code, we need to guarantee that the changes have been performed
        // before the handler is triggered.
        compiler_fence(Release);
    }

    // Ensure that the key is always non-null. Since key values are below
    // 1088, this cannot overflow.
    key + 1
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    let r = c::TlsSetValue(key - 1, value as c::LPVOID);
    debug_assert!(r != 0);
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    c::TlsGetValue(key - 1) as *mut u8
}

#[inline]
pub unsafe fn destroy(key: Key) {
    DTORS[(key - 1) as usize].store(ptr::null_mut(), Relaxed);
    let r = c::TlsFree(key - 1);
    // Use release ordering for the same reason as above.
    compiler_fence(Release);
    debug_assert!(r != 0);
}

#[allow(dead_code)] // actually called below
unsafe fn run_dtors() {
    let mut iterations = 5;
    while iterations != 0 {
        let mut any_run = false;
        // All keys have either been created by the current thread or must
        // have been propagated through other means of synchronization, so
        // we can just use relaxed ordering here and still observe all
        // changes relevant to us.
        let highest = HIGHEST.load(Relaxed);
        for (index, dtor) in DTORS[..highest].iter().enumerate() {
            let dtor = mem::transmute::<*mut (), Option<Dtor>>(dtor.load(Relaxed));
            if let Some(dtor) = dtor {
                let ptr = c::TlsGetValue(index as Key) as *mut u8;
                if !ptr.is_null() {
                    let r = c::TlsSetValue(index as Key, ptr::null_mut());
                    debug_assert!(r != 0);

                    (dtor)(ptr);
                    any_run = true;
                }
            }
        }

        iterations -= 1;
        // If no destructors where run, no new keys have been initialized,
        // so we are done. FIXME: Maybe use TLS to store the number of active
        // keys per thread.
        if !any_run {
            return;
        }
    }
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
    if dwReason == c::DLL_THREAD_DETACH || dwReason == c::DLL_PROCESS_DETACH {
        run_dtors();
        #[cfg(target_thread_local)]
        super::thread_local_dtor::run_keyless_dtors();
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
