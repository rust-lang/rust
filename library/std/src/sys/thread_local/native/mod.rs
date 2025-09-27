//! Thread local support for platforms with native TLS.
//!
//! To achieve the best performance, we choose from four different types for
//! the TLS variable, depending on the method of initialization used (`const`
//! or lazy) and the drop requirements of the stored type:
//!
//! |         | `Drop`               | `!Drop`             |
//! |--------:|:--------------------:|:-------------------:|
//! | `const` | `EagerStorage<T>`    | `T`                 |
//! | lazy    | `LazyStorage<T, ()>` | `LazyStorage<T, !>` |
//!
//! For `const` initialization and `!Drop` types, we simply use `T` directly,
//! but for other situations, we implement a state machine to handle
//! initialization of the variable and its destructor and destruction.
//! Upon accessing the TLS variable, the current state is compared:
//!
//! 1. If the state is `Initial`, initialize the storage, transition the state
//!    to `Alive` and (if applicable) register the destructor, and return a
//!    reference to the value.
//! 2. If the state is `Alive`, initialization was previously completed, so
//!    return a reference to the value.
//! 3. If the state is `Destroyed`, the destructor has been run already, so
//!    return [`None`].
//!
//! The TLS destructor sets the state to `Destroyed` and drops the current value.
//!
//! To simplify the code, we make `LazyStorage` generic over the destroyed state
//! and use the `!` type (never type) as type parameter for `!Drop` types. This
//! eliminates the `Destroyed` state for these values, which can allow more niche
//! optimizations to occur for the `State` enum. For `Drop` types, `()` is used.

use crate::cell::Cell;
use crate::ptr;

mod eager;
mod lazy;

pub use eager::Storage as EagerStorage;
pub use lazy::Storage as LazyStorage;

#[doc(hidden)]
#[allow_internal_unstable(
    thread_local_internals,
    cfg_target_thread_local,
    thread_local,
    never_type
)]
#[allow_internal_unsafe]
#[unstable(feature = "thread_local_internals", issue = "none")]
#[rustc_macro_transparency = "semitransparent"]
pub macro thread_local_inner {
    // NOTE: we cannot import `LocalKey`, `LazyStorage` or `EagerStorage` with a `use` because that
    // can shadow user provided type or type alias with a matching name. Please update the shadowing
    // test in `tests/thread.rs` if these types are renamed.

    // Used to generate the `LocalKey` value for const-initialized thread locals.
    (@key $t:ty, $(#[$align_attr:meta])*, const $init:expr) => {{
        const __INIT: $t = $init;

        unsafe {
            $crate::thread::LocalKey::new(const {
                if $crate::mem::needs_drop::<$t>() {
                    |_| {
                        #[thread_local]
                        $(#[$align_attr])*
                        static VAL: $crate::thread::local_impl::EagerStorage<$t>
                            = $crate::thread::local_impl::EagerStorage::new(__INIT);
                        VAL.get()
                    }
                } else {
                    |_| {
                        #[thread_local]
                        $(#[$align_attr])*
                        static VAL: $t = __INIT;
                        &VAL
                    }
                }
            })
        }
    }},

    // used to generate the `LocalKey` value for `thread_local!`
    (@key $t:ty, $(#[$align_attr:meta])*, $init:expr) => {{
        #[inline]
        fn __init() -> $t {
            $init
        }

        unsafe {
            $crate::thread::LocalKey::new(const {
                if $crate::mem::needs_drop::<$t>() {
                    |init| {
                        #[thread_local]
                        $(#[$align_attr])*
                        static VAL: $crate::thread::local_impl::LazyStorage<$t, ()>
                            = $crate::thread::local_impl::LazyStorage::new();
                        VAL.get_or_init(init, __init)
                    }
                } else {
                    |init| {
                        #[thread_local]
                        $(#[$align_attr])*
                        static VAL: $crate::thread::local_impl::LazyStorage<$t, !>
                            = $crate::thread::local_impl::LazyStorage::new();
                        VAL.get_or_init(init, __init)
                    }
                }
            })
        }
    }},
}

#[rustc_macro_transparency = "semitransparent"]
pub(crate) macro local_pointer {
    () => {},
    ($vis:vis static $name:ident; $($rest:tt)*) => {
        #[thread_local]
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
