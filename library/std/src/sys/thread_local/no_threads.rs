//! On some targets like wasm there's no threads, so no need to generate
//! thread locals and we can instead just use plain statics!

use crate::cell::{Cell, UnsafeCell};
use crate::mem::MaybeUninit;
use crate::ptr;

#[doc(hidden)]
#[allow_internal_unstable(thread_local_internals)]
#[allow_internal_unsafe]
#[unstable(feature = "thread_local_internals", issue = "none")]
#[rustc_macro_transparency = "semitransparent"]
pub macro thread_local_inner {
    // used to generate the `LocalKey` value for const-initialized thread locals
    (@key $t:ty, $(#[$align_attr:meta])*, const $init:expr) => {{
        const __INIT: $t = $init;

        // NOTE: Please update the shadowing test in `tests/thread.rs` if these types are renamed.
        unsafe {
            $crate::thread::LocalKey::new(|_| {
                $(#[$align_attr])*
                static VAL: $crate::thread::local_impl::EagerStorage<$t> =
                    $crate::thread::local_impl::EagerStorage { value: __INIT };
                &VAL.value
            })
        }
    }},

    // used to generate the `LocalKey` value for `thread_local!`
    (@key $t:ty, $(#[$align_attr:meta])*, $init:expr) => {{
        #[inline]
        fn __init() -> $t { $init }

        unsafe {
            $crate::thread::LocalKey::new(|init| {
                $(#[$align_attr])*
                static VAL: $crate::thread::local_impl::LazyStorage<$t> = $crate::thread::local_impl::LazyStorage::new();
                VAL.get(init, __init)
            })
        }
    }},
}

#[allow(missing_debug_implementations)]
#[repr(transparent)] // Required for correctness of `#[rustc_align_static]`
pub struct EagerStorage<T> {
    pub value: T,
}

// SAFETY: the target doesn't have threads.
unsafe impl<T> Sync for EagerStorage<T> {}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Initial,
    Alive,
    Destroying,
}

#[allow(missing_debug_implementations)]
#[repr(C)]
pub struct LazyStorage<T> {
    // This field must be first, for correctness of `#[rustc_align_static]`
    value: UnsafeCell<MaybeUninit<T>>,
    state: Cell<State>,
}

impl<T> LazyStorage<T> {
    pub const fn new() -> LazyStorage<T> {
        LazyStorage {
            value: UnsafeCell::new(MaybeUninit::uninit()),
            state: Cell::new(State::Initial),
        }
    }

    /// Gets a pointer to the TLS value, potentially initializing it with the
    /// provided parameters.
    ///
    /// The resulting pointer may not be used after reentrant inialialization
    /// has occurred.
    #[inline]
    pub fn get(&'static self, i: Option<&mut Option<T>>, f: impl FnOnce() -> T) -> *const T {
        if self.state.get() == State::Alive {
            self.value.get() as *const T
        } else {
            self.initialize(i, f)
        }
    }

    #[cold]
    fn initialize(&'static self, i: Option<&mut Option<T>>, f: impl FnOnce() -> T) -> *const T {
        let value = i.and_then(Option::take).unwrap_or_else(f);

        // Destroy the old value if it is initialized
        // FIXME(#110897): maybe panic on recursive initialization.
        if self.state.get() == State::Alive {
            self.state.set(State::Destroying);
            // Safety: we check for no initialization during drop below
            unsafe {
                ptr::drop_in_place(self.value.get() as *mut T);
            }
            self.state.set(State::Initial);
        }

        // Guard against initialization during drop
        if self.state.get() == State::Destroying {
            panic!("Attempted to initialize thread-local while it is being dropped");
        }

        unsafe {
            self.value.get().write(MaybeUninit::new(value));
        }
        self.state.set(State::Alive);

        self.value.get() as *const T
    }
}

// SAFETY: the target doesn't have threads.
unsafe impl<T> Sync for LazyStorage<T> {}

#[rustc_macro_transparency = "semitransparent"]
pub(crate) macro local_pointer {
    () => {},
    ($vis:vis static $name:ident; $($rest:tt)*) => {
        $vis static $name: $crate::sys::thread_local::LocalPointer = $crate::sys::thread_local::LocalPointer::__new();
        $crate::sys::thread_local::local_pointer! { $($rest)* }
    },
}

pub(crate) struct LocalPointer {
    p: Cell<*mut ()>,
}

impl LocalPointer {
    pub const fn __new() -> LocalPointer {
        LocalPointer { p: Cell::new(ptr::null_mut()) }
    }

    pub fn get(&self) -> *mut () {
        self.p.get()
    }

    pub fn set(&self, p: *mut ()) {
        self.p.set(p)
    }
}

// SAFETY: the target doesn't have threads.
unsafe impl Sync for LocalPointer {}
