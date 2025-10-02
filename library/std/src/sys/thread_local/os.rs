use super::key::{Key, LazyKey, get, set};
use super::{abort_on_dtor_unwind, guard};
use crate::alloc::{self, Layout};
use crate::cell::Cell;
use crate::marker::PhantomData;
use crate::mem::ManuallyDrop;
use crate::ops::Deref;
use crate::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use crate::ptr::{self, NonNull};

#[doc(hidden)]
#[allow_internal_unstable(thread_local_internals)]
#[allow_internal_unsafe]
#[unstable(feature = "thread_local_internals", issue = "none")]
#[rustc_macro_transparency = "semitransparent"]
pub macro thread_local_inner {
    // NOTE: we cannot import `Storage` or `LocalKey` with a `use` because that can shadow user
    // provided type or type alias with a matching name. Please update the shadowing test in
    // `tests/thread.rs` if these types are renamed.

    // used to generate the `LocalKey` value for `thread_local!`.
    (@key $t:ty, $($(#[$($align_attr:tt)*])+)?, $init:expr) => {{
        #[inline]
        fn __init() -> $t { $init }

        // NOTE: this cannot import `LocalKey` or `Storage` with a `use` because that can shadow
        // user provided type or type alias with a matching name. Please update the shadowing test
        // in `tests/thread.rs` if these types are renamed.
        unsafe {
            $crate::thread::LocalKey::new(|init| {
                static VAL: $crate::thread::local_impl::Storage<$t, {
                    $({
                        // Ensure that attributes have valid syntax
                        // and that the proper feature gate is enabled
                        $(#[$($align_attr)*])+
                        #[allow(unused)]
                        static DUMMY: () = ();
                    })?

                    #[allow(unused_mut)]
                    let mut final_align = $crate::thread::local_impl::value_align::<$t>();
                    $($($crate::thread::local_impl::thread_local_inner!(@align final_align, $($align_attr)*);)+)?
                    final_align
                }>
                    = $crate::thread::local_impl::Storage::new();
                VAL.get(init, __init)
            })
        }
    }},

    // process a single `rustc_align_static` attribute
    (@align $final_align:ident, rustc_align_static($($align:tt)*) $(, $($attr_rest:tt)+)?) => {
        let new_align: $crate::primitive::usize = $($align)*;
        if new_align > $final_align {
            $final_align = new_align;
        }

        $($crate::thread::local_impl::thread_local_inner!(@align $final_align, $($attr_rest)+);)?
    },

    // process a single `cfg_attr` attribute
    // by translating it into a `cfg`ed block and recursing.
    // https://doc.rust-lang.org/reference/conditional-compilation.html#railroad-ConfigurationPredicate

    (@align $final_align:ident, cfg_attr(true, $($cfg_rhs:tt)*) $(, $($attr_rest:tt)+)?) => {
        #[cfg(true)]
        {
            $crate::thread::local_impl::thread_local_inner!(@align $final_align, $($cfg_rhs)*);
        }

        $($crate::thread::local_impl::thread_local_inner!(@align $final_align, $($attr_rest)+);)?
    },

    (@align $final_align:ident, cfg_attr(false, $($cfg_rhs:tt)*) $(, $($attr_rest:tt)+)?) => {
        #[cfg(false)]
        {
            $crate::thread::local_impl::thread_local_inner!(@align $final_align, $($cfg_rhs)*);
        }

        $($crate::thread::local_impl::thread_local_inner!(@align $final_align, $($attr_rest)+);)?
    },

    (@align $final_align:ident, cfg_attr($cfg_pred:meta, $($cfg_rhs:tt)*) $(, $($attr_rest:tt)+)?) => {
        #[cfg($cfg_pred)]
        {
            $crate::thread::local_impl::thread_local_inner!(@align $final_align, $($cfg_rhs)*);
        }

        $($crate::thread::local_impl::thread_local_inner!(@align $final_align, $($attr_rest)+);)?
    },
}

/// Use a regular global static to store this key; the state provided will then be
/// thread-local.
/// INVARIANT: ALIGN must be a valid alignment, and no less than `value_align::<T>`.
#[allow(missing_debug_implementations)]
pub struct Storage<T, const ALIGN: usize> {
    key: LazyKey,
    marker: PhantomData<Cell<T>>,
}

unsafe impl<T, const ALIGN: usize> Sync for Storage<T, ALIGN> {}

#[repr(C)]
struct Value<T: 'static> {
    // This field must be first, for correctness of `#[rustc_align_static]`
    value: T,
    // INVARIANT: if this value is stored under a TLS key, `key` must be that `key`.
    key: Key,
}

pub const fn value_align<T: 'static>() -> usize {
    crate::mem::align_of::<Value<T>>()
}

/// Equivalent to `Box<Value<T>>`, but potentially over-aligned.
struct AlignedBox<T: 'static, const ALIGN: usize> {
    ptr: NonNull<Value<T>>,
}

impl<T: 'static, const ALIGN: usize> AlignedBox<T, ALIGN> {
    #[inline]
    fn new(v: Value<T>) -> Self {
        let layout = Layout::new::<Value<T>>().align_to(ALIGN).unwrap();

        let ptr: *mut Value<T> = (unsafe { alloc::alloc(layout) }).cast();
        let Some(ptr) = NonNull::new(ptr) else {
            alloc::handle_alloc_error(layout);
        };
        unsafe { ptr.write(v) };
        Self { ptr }
    }

    #[inline]
    fn into_raw(b: Self) -> *mut Value<T> {
        let md = ManuallyDrop::new(b);
        md.ptr.as_ptr()
    }

    #[inline]
    unsafe fn from_raw(ptr: *mut Value<T>) -> Self {
        Self { ptr: unsafe { NonNull::new_unchecked(ptr) } }
    }
}

impl<T: 'static, const ALIGN: usize> Deref for AlignedBox<T, ALIGN> {
    type Target = Value<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self.ptr.as_ptr()) }
    }
}

impl<T: 'static, const ALIGN: usize> Drop for AlignedBox<T, ALIGN> {
    #[inline]
    fn drop(&mut self) {
        let layout = Layout::new::<Value<T>>().align_to(ALIGN).unwrap();

        unsafe {
            let unwind_result = catch_unwind(AssertUnwindSafe(|| self.ptr.drop_in_place()));
            alloc::dealloc(self.ptr.as_ptr().cast(), layout);
            if let Err(payload) = unwind_result {
                resume_unwind(payload);
            }
        }
    }
}

impl<T: 'static, const ALIGN: usize> Storage<T, ALIGN> {
    pub const fn new() -> Storage<T, ALIGN> {
        Storage { key: LazyKey::new(Some(destroy_value::<T, ALIGN>)), marker: PhantomData }
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

        let value = AlignedBox::<T, ALIGN>::new(Value {
            value: i.and_then(Option::take).unwrap_or_else(f),
            key,
        });
        let ptr = AlignedBox::into_raw(value);

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
            drop(unsafe { AlignedBox::<T, ALIGN>::from_raw(old) });
        }

        // SAFETY: We just created this value above.
        unsafe { &(*ptr).value }
    }
}

unsafe extern "C" fn destroy_value<T: 'static, const ALIGN: usize>(ptr: *mut u8) {
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
        let ptr = unsafe { AlignedBox::<T, ALIGN>::from_raw(ptr as *mut Value<T>) };
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
