use crate::ffi::{CStr, c_char, c_void};
use crate::marker::{FnPtr, PhantomData};
use crate::sync::atomic::{Atomic, AtomicPtr, Ordering};
use crate::{mem, ptr};

#[cfg(test)]
#[path = "./tests.rs"]
mod tests;

pub(crate) macro weak {
    (fn $name:ident($($param:ident : $t:ty),* $(,)?) -> $ret:ty;) => (
        static DLSYM: DlsymWeak<unsafe extern "C" fn($($t),*) -> $ret> = {
            let Ok(name) = CStr::from_bytes_with_nul(concat!(stringify!($name), '\0').as_bytes()) else {
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
    pub const unsafe fn new(name: &'static CStr) -> Self {
        DlsymWeak {
            name: name.as_ptr(),
            func: AtomicPtr::new(ptr::without_provenance_mut(1)),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn get(&self) -> Option<F> {
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
            // the `weak!` macro delegates this responsibility to the caller
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
