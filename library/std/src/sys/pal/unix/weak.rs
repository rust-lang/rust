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
#![forbid(unsafe_op_in_unsafe_fn)]

use crate::ffi::{CStr, c_char, c_void};
use crate::marker::{FnPtr, PhantomData};
use crate::sync::atomic::{Atomic, AtomicPtr, Ordering};
use crate::{mem, ptr};

// We currently only test `dlsym!`, but that doesn't work on all platforms, so
// we gate the tests to only the platforms where it is actually used.
//
// FIXME(joboet): add more tests, reorganise the whole module and get rid of
//                `#[allow(dead_code, unused_macros)]`.
#[cfg(any(
    target_vendor = "apple",
    all(target_os = "linux", target_env = "gnu"),
    target_os = "freebsd",
))]
#[cfg(test)]
mod tests;

// We can use true weak linkage on ELF targets.
#[cfg(all(unix, not(target_vendor = "apple")))]
pub(crate) macro weak {
    (fn $name:ident($($param:ident : $t:ty),* $(,)?) -> $ret:ty;) => (
        let ref $name: ExternWeak<unsafe extern "C" fn($($t),*) -> $ret> = {
            unsafe extern "C" {
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
    (fn $name:ident($($param:ident : $t:ty),* $(,)?) -> $ret:ty;) => (
        dlsym!(
            #[link_name = stringify!($name)]
            fn $name($($param : $t),*) -> $ret;
        );
    ),
    (
        #[link_name = $sym:expr]
        fn $name:ident($($param:ident : $t:ty),* $(,)?) -> $ret:ty;
    ) => (
        static DLSYM: DlsymWeak<unsafe extern "C" fn($($t),*) -> $ret> = {
            let Ok(name) = CStr::from_bytes_with_nul(concat!($sym, '\0').as_bytes()) else {
                panic!("symbol name may not contain NUL")
            };

            // SAFETY: Whoever calls the function pointer returned by `get()`
            // is responsible for ensuring that the signature is correct. Just
            // like with extern blocks, this is syntactically enforced by making
            // the function pointer be unsafe.
            unsafe { DlsymWeak::new(name) }
        };

        let $name = &DLSYM;
    )
}

pub(crate) struct DlsymWeak<F> {
    /// A pointer to the nul-terminated name of the symbol.
    // Use a pointer instead of `&'static CStr` to save space.
    name: *const c_char,
    func: Atomic<*mut libc::c_void>,
    _marker: PhantomData<F>,
}

impl<F: FnPtr> DlsymWeak<F> {
    /// # Safety
    ///
    /// If the signature of `F` does not match the signature of the symbol (if
    /// it exists), calling the function pointer returned by `get()` is
    /// undefined behaviour.
    pub(crate) const unsafe fn new(name: &'static CStr) -> Self {
        DlsymWeak {
            name: name.as_ptr(),
            func: AtomicPtr::new(ptr::without_provenance_mut(1)),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn get(&self) -> Option<F> {
        // The caller is presumably going to read through this value
        // (by calling the function we've dlsymed). This means we'd
        // need to have loaded it with at least C11's consume
        // ordering in order to be guaranteed that the data we read
        // from the pointer isn't from before the pointer was
        // stored. Rust has no equivalent to memory_order_consume,
        // so we use an acquire load (sorry, ARM).
        //
        // Now, in practice this likely isn't needed even on CPUs
        // where relaxed and consume mean different things. The
        // symbols we're loading are probably present (or not) at
        // init, and even if they aren't the runtime dynamic loader
        // is extremely likely have sufficient barriers internally
        // (possibly implicitly, for example the ones provided by
        // invoking `mprotect`).
        //
        // That said, none of that's *guaranteed*, so we use acquire.
        match self.func.load(Ordering::Acquire) {
            func if func.addr() == 1 => self.initialize(),
            func if func.is_null() => None,
            // SAFETY:
            // `func` is not null and `F` implements `FnPtr`, thus this
            // transmutation is well-defined. It is the responsibility of the
            // creator of this `DlsymWeak` to ensure that calling the resulting
            // function pointer does not result in undefined behaviour (though
            // the `dlsym!` macro delegates this responsibility to the caller
            // of the function by using `unsafe` function pointers).
            // FIXME: use `transmute` once it stops complaining about generics.
            func => Some(unsafe { mem::transmute_copy::<*mut c_void, F>(&func) }),
        }
    }

    // Cold because it should only happen during first-time initialization.
    #[cold]
    fn initialize(&self) -> Option<F> {
        // SAFETY: `self.name` was created from a `&'static CStr` and is
        // therefore a valid C string pointer.
        let val = unsafe { libc::dlsym(libc::RTLD_DEFAULT, self.name) };
        // This synchronizes with the acquire load in `get`.
        self.func.store(val, Ordering::Release);

        if val.is_null() {
            None
        } else {
            // SAFETY: see the comment in `get`.
            // FIXME: use `transmute` once it stops complaining about generics.
            Some(unsafe { mem::transmute_copy::<*mut libc::c_void, F>(&val) })
        }
    }
}

unsafe impl<F> Send for DlsymWeak<F> {}
unsafe impl<F> Sync for DlsymWeak<F> {}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub(crate) macro syscall {
    (fn $name:ident($($param:ident : $t:ty),* $(,)?) -> $ret:ty;) => (
        unsafe fn $name($($param: $t),*) -> $ret {
            weak!(fn $name($($param: $t),*) -> $ret;);

            if let Some(fun) = $name.get() {
                unsafe { fun($($param),*) }
            } else {
                super::os::set_errno(libc::ENOSYS);
                -1
            }
        }
    )
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) macro syscall {
    (
        fn $name:ident($($param:ident : $t:ty),* $(,)?) -> $ret:ty;
    ) => (
        unsafe fn $name($($param: $t),*) -> $ret {
            weak!(fn $name($($param: $t),*) -> $ret;);

            // Use a weak symbol from libc when possible, allowing `LD_PRELOAD`
            // interposition, but if it's not found just use a raw syscall.
            if let Some(fun) = $name.get() {
                unsafe { fun($($param),*) }
            } else {
                unsafe { libc::syscall(libc::${concat(SYS_, $name)}, $($param),*) as $ret }
            }
        }
    )
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) macro raw_syscall {
    (fn $name:ident($($param:ident : $t:ty),* $(,)?) -> $ret:ty;) => (
        unsafe fn $name($($param: $t),*) -> $ret {
            unsafe { libc::syscall(libc::${concat(SYS_, $name)}, $($param),*) as $ret }
        }
    )
}
