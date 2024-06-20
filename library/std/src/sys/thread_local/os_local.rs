use super::abort_on_dtor_unwind;
use crate::cell::Cell;
use crate::marker::PhantomData;
use crate::ptr;
use crate::sys_common::thread_local_key::StaticKey as OsKey;

#[doc(hidden)]
#[allow_internal_unstable(thread_local_internals)]
#[allow_internal_unsafe]
#[unstable(feature = "thread_local_internals", issue = "none")]
#[rustc_macro_transparency = "semitransparent"]
pub macro thread_local_inner {
    // used to generate the `LocalKey` value for const-initialized thread locals
    (@key $t:ty, const $init:expr) => {
        $crate::thread::local_impl::thread_local_inner!(@key $t, { const INIT_EXPR: $t = $init; INIT_EXPR })
    },

    // used to generate the `LocalKey` value for `thread_local!`
    (@key $t:ty, $init:expr) => {{
        #[inline]
        fn __init() -> $t { $init }

        unsafe {
            use $crate::thread::LocalKey;
            use $crate::thread::local_impl::Key;

            // Inlining does not work on windows-gnu due to linking errors around
            // dllimports. See https://github.com/rust-lang/rust/issues/109797.
            LocalKey::new(#[cfg_attr(windows, inline(never))] |init| {
                static VAL: Key<$t> = Key::new();
                VAL.get(init, __init)
            })
        }
    }},
    ($(#[$attr:meta])* $vis:vis $name:ident, $t:ty, $($init:tt)*) => {
        $(#[$attr])* $vis const $name: $crate::thread::LocalKey<$t> =
            $crate::thread::local_impl::thread_local_inner!(@key $t, $($init)*);
    },
}

/// Use a regular global static to store this key; the state provided will then be
/// thread-local.
#[allow(missing_debug_implementations)]
pub struct Key<T> {
    os: OsKey,
    marker: PhantomData<Cell<T>>,
}

unsafe impl<T> Sync for Key<T> {}

struct Value<T: 'static> {
    value: T,
    key: &'static Key<T>,
}

impl<T: 'static> Key<T> {
    #[rustc_const_unstable(feature = "thread_local_internals", issue = "none")]
    pub const fn new() -> Key<T> {
        Key { os: OsKey::new(Some(destroy_value::<T>)), marker: PhantomData }
    }

    /// Get a pointer to the TLS value, potentially initializing it with the
    /// provided parameters. If the TLS variable has been destroyed, a null
    /// pointer is returned.
    ///
    /// The resulting pointer may not be used after reentrant inialialization
    /// or thread destruction has occurred.
    pub fn get(&'static self, i: Option<&mut Option<T>>, f: impl FnOnce() -> T) -> *const T {
        // SAFETY: (FIXME: get should actually be safe)
        let ptr = unsafe { self.os.get() as *mut Value<T> };
        if ptr.addr() > 1 {
            // SAFETY: the check ensured the pointer is safe (its destructor
            // is not running) + it is coming from a trusted source (self).
            unsafe { &(*ptr).value }
        } else {
            self.try_initialize(ptr, i, f)
        }
    }

    fn try_initialize(
        &'static self,
        ptr: *mut Value<T>,
        i: Option<&mut Option<T>>,
        f: impl FnOnce() -> T,
    ) -> *const T {
        if ptr.addr() == 1 {
            // destructor is running
            return ptr::null();
        }

        let value = i.and_then(Option::take).unwrap_or_else(f);
        let ptr = Box::into_raw(Box::new(Value { value, key: self }));
        // SAFETY: (FIXME: get should actually be safe)
        let old = unsafe { self.os.get() as *mut Value<T> };
        // SAFETY: `ptr` is a correct pointer that can be destroyed by the key destructor.
        unsafe {
            self.os.set(ptr as *mut u8);
        }
        if !old.is_null() {
            // If the variable was recursively initialized, drop the old value.
            // SAFETY: We cannot be inside a `LocalKey::with` scope, as the
            // initializer has already returned and the next scope only starts
            // after we return the pointer. Therefore, there can be no references
            // to the old value.
            drop(unsafe { Box::from_raw(old) });
        }

        // SAFETY: We just created this value above.
        unsafe { &(*ptr).value }
    }
}

unsafe extern "C" fn destroy_value<T: 'static>(ptr: *mut u8) {
    // SAFETY:
    //
    // The OS TLS ensures that this key contains a null value when this
    // destructor starts to run. We set it back to a sentinel value of 1 to
    // ensure that any future calls to `get` for this thread will return
    // `None`.
    //
    // Note that to prevent an infinite loop we reset it back to null right
    // before we return from the destructor ourselves.
    abort_on_dtor_unwind(|| {
        let ptr = unsafe { Box::from_raw(ptr as *mut Value<T>) };
        let key = ptr.key;
        unsafe { key.os.set(ptr::without_provenance_mut(1)) };
        drop(ptr);
        unsafe { key.os.set(ptr::null_mut()) };
    });
}
