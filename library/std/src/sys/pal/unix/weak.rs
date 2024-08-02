//! Support for "weak linkage" to symbols on Unix
//!
//! Some I/O operations we do in std require newer versions of OSes but we need
//! to maintain binary compatibility with older releases for now. In order to
//! use the new functionality when available we use this module for detection.
//!
//! One option to use here is weak linkage, but that is unfortunately only
//! really workable with ELF. Otherwise, use dlsym to get the symbol value at
//! runtime. This is also done for compatibility with older versions of glibc,
//! and to avoid creating dependencies on GLIBC_PRIVATE symbols. It assumes that
//! we've been dynamically linked to the library the symbol comes from, but that
//! is currently always the case for things like libpthread/libc.
//!
//! A long time ago this used weak linkage for the __pthread_get_minstack
//! symbol, but that caused Debian to detect an unnecessarily strict versioned
//! dependency on libc6 (#23628) because it is GLIBC_PRIVATE. We now use `dlsym`
//! for a runtime lookup of that symbol to avoid the ELF versioned dependency.

// There are a variety of `#[cfg]`s controlling which targets are involved in
// each instance of `weak!` and `syscall!`. Rather than trying to unify all of
// that, we'll just allow that some unix targets don't use this module at all.
#![allow(dead_code, unused_macros)]

use crate::ffi::CStr;
use crate::marker::PhantomData;
use crate::sync::atomic::{self, AtomicPtr, Ordering};
use crate::{mem, ptr};

// We can use true weak linkage on ELF targets.
#[cfg(all(unix, not(target_vendor = "apple")))]
pub(crate) macro weak {
    (fn $name:ident($($t:ty),*) -> $ret:ty) => (
        let ref $name: ExternWeak<unsafe extern "C" fn($($t),*) -> $ret> = {
            extern "C" {
                #[linkage = "extern_weak"]
                static $name: Option<unsafe extern "C" fn($($t),*) -> $ret>;
            }
            #[allow(unused_unsafe)]
            ExternWeak::new(unsafe { $name })
        };
    )
}

// On non-ELF targets, use the dlsym approximation of weak linkage.
#[cfg(target_vendor = "apple")]
pub(crate) use self::dlsym as weak;

pub(crate) struct ExternWeak<F: Copy> {
    weak_ptr: Option<F>,
}

impl<F: Copy> ExternWeak<F> {
    #[inline]
    pub(crate) fn new(weak_ptr: Option<F>) -> Self {
        ExternWeak { weak_ptr }
    }

    #[inline]
    pub(crate) fn get(&self) -> Option<F> {
        self.weak_ptr
    }
}

pub(crate) macro dlsym {
    (fn $name:ident($($t:ty),*) -> $ret:ty) => (
         dlsym!(fn $name($($t),*) -> $ret, stringify!($name));
    ),
    (fn $name:ident($($t:ty),*) -> $ret:ty, $sym:expr) => (
        static DLSYM: DlsymWeak<unsafe extern "C" fn($($t),*) -> $ret> =
            DlsymWeak::new(concat!($sym, '\0'));
        let $name = &DLSYM;
    )
}
pub(crate) struct DlsymWeak<F> {
    name: &'static str,
    func: AtomicPtr<libc::c_void>,
    _marker: PhantomData<F>,
}

impl<F> DlsymWeak<F> {
    pub(crate) const fn new(name: &'static str) -> Self {
        DlsymWeak {
            name,
            func: AtomicPtr::new(ptr::without_provenance_mut(1)),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn get(&self) -> Option<F> {
        unsafe {
            // Relaxed is fine here because we fence before reading through the
            // pointer (see the comment below).
            match self.func.load(Ordering::Relaxed) {
                func if func.addr() == 1 => self.initialize(),
                func if func.is_null() => None,
                func => {
                    let func = mem::transmute_copy::<*mut libc::c_void, F>(&func);
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

    // Cold because it should only happen during first-time initialization.
    #[cold]
    unsafe fn initialize(&self) -> Option<F> {
        assert_eq!(mem::size_of::<F>(), mem::size_of::<*mut libc::c_void>());

        let val = fetch(self.name);
        // This synchronizes with the acquire fence in `get`.
        self.func.store(val, Ordering::Release);

        if val.is_null() { None } else { Some(mem::transmute_copy::<*mut libc::c_void, F>(&val)) }
    }
}

unsafe fn fetch(name: &str) -> *mut libc::c_void {
    let name = match CStr::from_bytes_with_nul(name.as_bytes()) {
        Ok(cstr) => cstr,
        Err(..) => return ptr::null_mut(),
    };
    libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr())
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub(crate) macro syscall {
    (fn $name:ident($($arg_name:ident: $t:ty),*) -> $ret:ty) => (
        unsafe fn $name($($arg_name: $t),*) -> $ret {
            weak! { fn $name($($t),*) -> $ret }

            if let Some(fun) = $name.get() {
                fun($($arg_name),*)
            } else {
                super::os::set_errno(libc::ENOSYS);
                -1
            }
        }
    )
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) macro syscall {
    (fn $name:ident($($arg_name:ident: $t:ty),*) -> $ret:ty) => (
        unsafe fn $name($($arg_name:$t),*) -> $ret {
            weak! { fn $name($($t),*) -> $ret }

            // Use a weak symbol from libc when possible, allowing `LD_PRELOAD`
            // interposition, but if it's not found just use a raw syscall.
            if let Some(fun) = $name.get() {
                fun($($arg_name),*)
            } else {
                libc::syscall(libc::${concat(SYS_, $name)}, $($arg_name),*) as $ret
            }
        }
    )
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) macro raw_syscall {
    (fn $name:ident($($arg_name:ident: $t:ty),*) -> $ret:ty) => (
        unsafe fn $name($($arg_name:$t),*) -> $ret {
            libc::syscall(libc::${concat(SYS_, $name)}, $($arg_name),*) as $ret
        }
    )
}
