//! Support for "weak linkage" to symbols on Unix
//!
//! Some I/O operations we do in libstd require newer versions of OSes but we
//! need to maintain binary compatibility with older releases for now. In order
//! to use the new functionality when available we use this module for
//! detection.
//!
//! One option to use here is weak linkage, but that is unfortunately only
//! really workable on Linux. Hence, use dlsym to get the symbol value at
//! runtime. This is also done for compatibility with older versions of glibc,
//! and to avoid creating dependencies on GLIBC_PRIVATE symbols. It assumes that
//! we've been dynamically linked to the library the symbol comes from, but that
//! is currently always the case for things like libpthread/libc.
//!
//! A long time ago this used weak linkage for the __pthread_get_minstack
//! symbol, but that caused Debian to detect an unnecessarily strict versioned
//! dependency on libc6 (#23628).

// There are a variety of `#[cfg]`s controlling which targets are involved in
// each instance of `weak!` and `syscall!`. Rather than trying to unify all of
// that, we'll just allow that some unix targets don't use this module at all.
#![allow(dead_code, unused_macros)]

use crate::ffi::CStr;
use crate::marker;
use crate::mem;
use crate::sync::atomic::{self, AtomicUsize, Ordering};

pub(crate) macro weak {
    (fn $name:ident($($t:ty),*) -> $ret:ty) => (
        weak!(fn $name($($t),*) -> $ret, stringify!($name));
    ),
    (fn $name:ident($($t:ty),*) -> $ret:ty, $sym:expr) => (
        #[allow(non_upper_case_globals)]
        static $name: crate::sys::weak::Weak<unsafe extern "C" fn($($t),*) -> $ret> =
            crate::sys::weak::Weak::new(concat!($sym, '\0'));
    )
}

pub struct Weak<F> {
    name: &'static str,
    addr: AtomicUsize,
    _marker: marker::PhantomData<F>,
}

impl<F> Weak<F> {
    pub const fn new(name: &'static str) -> Weak<F> {
        Weak { name, addr: AtomicUsize::new(1), _marker: marker::PhantomData }
    }

    pub fn get(&self) -> Option<F> {
        assert_eq!(mem::size_of::<F>(), mem::size_of::<usize>());
        unsafe {
            // Relaxed is fine here because we fence before reading through the
            // pointer (see the comment below).
            match self.addr.load(Ordering::Relaxed) {
                1 => self.initialize(),
                0 => None,
                addr => {
                    let func = mem::transmute_copy::<usize, F>(&addr);
                    // The caller is presumably going to read through this value
                    // (by calling the function we've dlsymed). This means we'd
                    // need to have loaded it with at least C11's consume
                    // ordering in order to be guaranteed that the data we read
                    // from the pointer isn't from before the pointer was
                    // stored. Rust has no equivalent to memory_order_consume,
                    // so we use an acquire fence (sorry, ARM).
                    //
                    // Now, in practice this likely isn't needed even on CPUs
                    // where relaxed and consume mean different things. The
                    // symbols we're loading are probably present (or not) at
                    // init, and even if they aren't the runtime dynamic loader
                    // is extremely likely have sufficient barriers internally
                    // (possibly implicitly, for example the ones provided by
                    // invoking `mprotect`).
                    //
                    // That said, none of that's *guaranteed*, and so we fence.
                    atomic::fence(Ordering::Acquire);
                    Some(func)
                }
            }
        }
    }

    // Cold because it should only happen during first-time initalization.
    #[cold]
    unsafe fn initialize(&self) -> Option<F> {
        let val = fetch(self.name);
        // This synchronizes with the acquire fence in `get`.
        self.addr.store(val, Ordering::Release);

        match val {
            0 => None,
            addr => Some(mem::transmute_copy::<usize, F>(&addr)),
        }
    }
}

unsafe fn fetch(name: &str) -> usize {
    let name = match CStr::from_bytes_with_nul(name.as_bytes()) {
        Ok(cstr) => cstr,
        Err(..) => return 0,
    };
    libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) as usize
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub(crate) macro syscall {
    (fn $name:ident($($arg_name:ident: $t:ty),*) -> $ret:ty) => (
        unsafe fn $name($($arg_name: $t),*) -> $ret {
            use super::os;

            weak! { fn $name($($t),*) -> $ret }

            if let Some(fun) = $name.get() {
                fun($($arg_name),*)
            } else {
                os::set_errno(libc::ENOSYS);
                -1
            }
        }
    )
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) macro syscall {
    (fn $name:ident($($arg_name:ident: $t:ty),*) -> $ret:ty) => (
        unsafe fn $name($($arg_name:$t),*) -> $ret {
            use weak;
            // This looks like a hack, but concat_idents only accepts idents
            // (not paths).
            use libc::*;

            weak! { fn $name($($t),*) -> $ret }

            // Use a weak symbol from libc when possible, allowing `LD_PRELOAD`
            // interposition, but if it's not found just use a raw syscall.
            if let Some(fun) = $name.get() {
                fun($($arg_name),*)
            } else {
                syscall(
                    concat_idents!(SYS_, $name),
                    $($arg_name),*
                ) as $ret
            }
        }
    )
}
