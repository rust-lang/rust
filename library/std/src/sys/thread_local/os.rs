use super::key::{Key, LazyKey, get, set};
use super::{abort_on_dtor_unwind, guard};
use crate::alloc::Layout;
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
    (@key $t:ty, $(#[$($align_attr:tt)*])*, const $init:expr) => {
        $crate::thread::local_impl::thread_local_inner!(@key $t, $(#[$($align_attr)*])*, { const INIT_EXPR: $t = $init; INIT_EXPR })
    },

    // NOTE: we cannot import `Storage` or `LocalKey` with a `use` because that can shadow user
    // provided type or type alias with a matching name. Please update the shadowing test in
    // `tests/thread.rs` if these types are renamed.

    // used to generate the `LocalKey` value for `thread_local!`.
    (@key $t:ty, $(#[$($align_attr:tt)*])*, $init:expr) => {{
        #[inline]
        fn __init() -> $t { $init }

        // NOTE: this cannot import `LocalKey` or `Storage` with a `use` because that can shadow
        // user provided type or type alias with a matching name. Please update the shadowing test
        // in `tests/thread.rs` if these types are renamed.
        unsafe {
            $crate::thread::LocalKey::new(|init| {
                static VAL: $crate::thread::local_impl::Storage<$t>
                    = $crate::thread::local_impl::Storage::new();
                VAL.get($crate::thread::local_impl::thread_local_inner!(@align $(#[$($align_attr)*])*), init, __init)
            })
        }
    }},

    // Handle `rustc_align_static` attributes,
    // by translating them into an argumemt to pass to `Storage::get`:

    // fast path for when there are none
    (@align) => (1),

    // `rustc_align_static` attributes are present,
    // translate them into a `const` block that computes the alignment
    (@align $(#[$($attr:tt)*])+) => {
        const {
            // Ensure that attributes have valid syntax
            // and that the proper feature gate is enabled
            $(#[$($attr)*])+
            static DUMMY: () = ();

            let mut final_align = 1_usize;
            $($crate::thread::local_impl::thread_local_inner!(@align_single final_align, $($attr)*);)+
            final_align
        }
    },

    // process a single `rustc_align_static` attribute
    (@align_single $final_align:ident, rustc_align_static $($attr_rest:tt)*) => {
        #[allow(unused_parens)]
        let new_align: usize = $($attr_rest)*;
        if new_align > $final_align {
            $final_align = new_align;
        }
    },

    // process a single `cfg_attr` attribute
    // by translating it into a `cfg`ed block and recursing.
    // https://doc.rust-lang.org/reference/conditional-compilation.html#railroad-ConfigurationPredicate

    (@align_single $final_align:ident, cfg_attr(true, $($cfg_rhs:tt)*)) => {
        #[cfg(true)]
        {
            $crate::thread::local_impl::thread_local_inner!(@align_single $final_align, $($cfg_rhs)*);
        }
    },

    (@align_single $final_align:ident, cfg_attr(false, $($cfg_rhs:tt)*)) => {
        #[cfg(false)]
        {
            $crate::thread::local_impl::thread_local_inner!(@align_single $final_align, $($cfg_rhs)*);
        }
    },

    (@align_single $final_align:ident, cfg_attr($cfg_op:ident ($($cfg_preds:tt)*), $($cfg_rhs:tt)*)) => {
        #[cfg($cfg_op ($($cfg_preds)*))]
        {
            $crate::thread::local_impl::thread_local_inner!(@align_single $final_align, $($cfg_rhs)*);
        }
    },

    (@align_single $final_align:ident, cfg_attr($cfg_ident:ident $(= $cfg_val:expr)?, $($cfg_rhs:tt)*)) => {
        #[cfg($cfg_ident $(= $cfg_val)?)]
        {
            $crate::thread::local_impl::thread_local_inner!(@align_single $final_align, $($cfg_rhs)*);
        }
    },


    ($(#[$attr:meta])* $vis:vis $name:ident, $t:ty, $(#[$($align_attr:tt)*])*, $($init:tt)*) => {
        $(#[$attr])* $vis const $name: $crate::thread::LocalKey<$t> =
            $crate::thread::local_impl::thread_local_inner!(@key $t, $(#[$($align_attr)*])*, $($init)*);
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

#[repr(C)]
struct Value<T: 'static> {
    // This field must be first, for correctness of `#[rustc_align_static]`
    value: T,
    align: usize,
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
    pub fn get(
        &'static self,
        align: usize,
        i: Option<&mut Option<T>>,
        f: impl FnOnce() -> T,
    ) -> *const T {
        let key = self.key.force();
        let ptr = unsafe { get(key) as *mut Value<T> };
        if ptr.addr() > 1 {
            // SAFETY: the check ensured the pointer is safe (its destructor
            // is not running) + it is coming from a trusted source (self).
            unsafe { &(*ptr).value }
        } else {
            // SAFETY: trivially correct.
            unsafe { Self::try_initialize(key, align, ptr, i, f) }
        }
    }

    /// # Safety
    /// * `key` must be the result of calling `self.key.force()`
    /// * `ptr` must be the current value associated with `key`.
    unsafe fn try_initialize(
        key: Key,
        align: usize,
        ptr: *mut Value<T>,
        i: Option<&mut Option<T>>,
        f: impl FnOnce() -> T,
    ) -> *const T {
        if ptr.addr() == 1 {
            // destructor is running
            return ptr::null();
        }

        // Manually allocate with the requested alignment
        let layout = Layout::new::<Value<T>>().align_to(align).unwrap();
        let ptr: *mut Value<T> = (unsafe { crate::alloc::alloc(layout) }).cast();
        if ptr.is_null() {
            crate::alloc::handle_alloc_error(layout);
        }
        unsafe {
            ptr.write(Value {
                value: i.and_then(Option::take).unwrap_or_else(f),
                align: layout.align(),
                key,
            });
        }

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
            unsafe {
                old.drop_in_place();
                crate::alloc::dealloc(old.cast(), layout);
            }
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
        let value_ptr: *mut Value<T> = ptr.cast();
        unsafe {
            let key = (*value_ptr).key;
            let align = (*value_ptr).align;

            // SAFETY: `key` is the TLS key `ptr` was stored under.
            set(key, ptr::without_provenance_mut(1));

            // drop and deallocate the value
            let layout =
                Layout::from_size_align_unchecked(crate::mem::size_of::<Value<T>>(), align);
            value_ptr.drop_in_place();
            crate::alloc::dealloc(ptr, layout);

            // SAFETY: `key` is the TLS key `ptr` was stored under.
            set(key, ptr::null_mut());
        };
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
