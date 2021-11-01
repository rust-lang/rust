use crate::mem::ManuallyDrop;
use crate::ptr;
use crate::sync::atomic::AtomicPtr;
use crate::sync::atomic::Ordering::SeqCst;
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
// To accomplish this feat, we perform a number of threads, all contained
// within this module:
//
// * All TLS destructors are tracked by *us*, not the windows runtime. This
//   means that we have a global list of destructors for each TLS key that
//   we know about.
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
// [1]: https://www.codeproject.com/Articles/8113/Thread-Local-Storage-The-C-Way
// [2]: https://github.com/ChromiumWebApps/chromium/blob/master/base
//                        /threading/thread_local_storage_win.cc#L42

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
    key
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
pub unsafe fn destroy(_key: Key) {
    rtabort!("can't destroy tls keys on windows")
}

#[inline]
pub fn requires_synchronized_create() -> bool {
    true
}

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
//
// Perhaps one day we can fold the `Box` here into a static allocation,
// expanding the `StaticKey` structure to contain not only a slot for the TLS
// key but also a slot for the destructor queue on windows. An optimization for
// another day!

static DTORS: AtomicPtr<Node> = AtomicPtr::new(ptr::null_mut());

struct Node {
    dtor: Dtor,
    key: Key,
    next: *mut Node,
}

unsafe fn register_dtor(key: Key, dtor: Dtor) {
    let mut node = ManuallyDrop::new(Box::new(Node { key, dtor, next: ptr::null_mut() }));

    let mut head = DTORS.load(SeqCst);
    loop {
        node.next = head;
        match DTORS.compare_exchange(head, &mut **node, SeqCst, SeqCst) {
            Ok(_) => return, // nothing to drop, we successfully added the node to the list
            Err(cur) => head = cur,
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

#[allow(dead_code)] // actually called above
unsafe fn run_dtors() {
    let mut any_run = true;
    for _ in 0..5 {
        if !any_run {
            break;
        }
        any_run = false;
        let mut cur = DTORS.load(SeqCst);
        while !cur.is_null() {
            let ptr = c::TlsGetValue((*cur).key);

            if !ptr.is_null() {
                c::TlsSetValue((*cur).key, ptr::null_mut());
                ((*cur).dtor)(ptr as *mut _);
                any_run = true;
            }

            cur = (*cur).next;
        }
    }
}
