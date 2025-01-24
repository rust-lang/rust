use super::key::{Key, LazyKey, get, set};
use super::{abort_on_dtor_unwind, guard};
use crate::cell::Cell;
use crate::marker::PhantomData;
use crate::ptr;

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

    // NOTE: we cannot import `Storage` or `LocalKey` with a `use` because that can shadow user
    // provided type or type alias with a matching name. Please update the shadowing test in
    // `tests/thread.rs` if these types are renamed.

    // used to generate the `LocalKey` value for `thread_local!`.
    (@key $t:ty, $init:expr) => {{
        #[inline]
        fn __init() -> $t { $init }

        // NOTE: this cannot import `LocalKey` or `Storage` with a `use` because that can shadow
        // user provided type or type alias with a matching name. Please update the shadowing test
        // in `tests/thread.rs` if these types are renamed.
        unsafe {
            // Inlining does not work on windows-gnu due to linking errors around
            // dllimports. See https://github.com/rust-lang/rust/issues/109797.
            $crate::thread::LocalKey::new(#[cfg_attr(windows, inline(never))] |init| {
                static VAL: $crate::thread::local_impl::Storage<$t>
                    = $crate::thread::local_impl::Storage::new();
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
pub struct Storage<T> {
    key: LazyKey,
    marker: PhantomData<Cell<T>>,
}

unsafe impl<T> Sync for Storage<T> {}

struct Value<T: 'static> {
    value: T,
    // INVARIANT: if this value is stored under a TLS key, `key` must be that `key`.
    key: Key,
}

impl<T: 'static> Storage<T> {
    pub const fn new() -> Storage<T> {
        Storage { key: LazyKey::new(Some(destroy_value::<T>)), marker: PhantomData }
    }

    /// Gets a pointer to the TLS value, potentially initializing it with the
    /// provided parameters. If the TLS variable has been destroyed, a null
    /// pointer is returned.
    ///
    /// The resulting pointer may not be used after reentrant inialialization
    /// or thread destruction has occurred.
    pub fn get(&'static self, i: Option<&mut Option<T>>, f: impl FnOnce() -> T) -> *const T {
        let key = self.key.force();
        let ptr = unsafe { get(key) as *mut Value<T> };
        if ptr.addr() > 1 {
            // SAFETY: the check ensured the pointer is safe (its destructor
            // is not running) + it is coming from a trusted source (self).
            unsafe { &(*ptr).value }
        } else {
            // SAFETY: trivially correct.
            unsafe { Self::try_initialize(key, ptr, i, f) }
        }
    }

    /// # Safety
    /// * `key` must be the result of calling `self.key.force()`
    /// * `ptr` must be the current value associated with `key`.
    unsafe fn try_initialize(
        key: Key,
        ptr: *mut Value<T>,
        i: Option<&mut Option<T>>,
        f: impl FnOnce() -> T,
    ) -> *const T {
        if ptr.addr() == 1 {
            // destructor is running
            return ptr::null();
        }

        let value = Box::new(Value { value: i.and_then(Option::take).unwrap_or_else(f), key });
        let ptr = Box::into_raw(value);

        // SAFETY:
        // * key came from a `LazyKey` and is thus correct.
        // * `ptr` is a correct pointer that can be destroyed by the key destructor.
        // * the value is stored under the key that it contains.
        let old = unsafe {
            let old = get(key) as *mut Value<T>;
            set(key, ptr as *mut u8);
            old
        };

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
        // SAFETY: `key` is the TLS key `ptr` was stored under.
        unsafe { set(key, ptr::without_provenance_mut(1)) };
        drop(ptr);
        // SAFETY: `key` is the TLS key `ptr` was stored under.
        unsafe { set(key, ptr::null_mut()) };
        // Make sure that the runtime cleanup will be performed
        // after the next round of TLS destruction.
        guard::enable();
    });
}

#[rustc_macro_transparency = "semitransparent"]
pub(crate) macro local_pointer {
    () => {},
    ($vis:vis static $name:ident; $($rest:tt)*) => {
        $vis static $name: $crate::sys::thread_local::LocalPointer = $crate::sys::thread_local::LocalPointer::__new();
        $crate::sys::thread_local::local_pointer! { $($rest)* }
    },
}

pub(crate) struct LocalPointer {
    key: LazyKey,
}

impl LocalPointer {
    pub const fn __new() -> LocalPointer {
        LocalPointer { key: LazyKey::new(None) }
    }

    pub fn get(&'static self) -> *mut () {
        unsafe { get(self.key.force()) as *mut () }
    }

    pub fn set(&'static self, p: *mut ()) {
        unsafe { set(self.key.force(), p as *mut u8) }
    }
}
