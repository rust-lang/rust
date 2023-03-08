//! Platform-dependent platform abstraction.
//!
//! The `std::sys` module is the abstracted interface through which
//! `std` talks to the underlying operating system. It has different
//! implementations for different operating system families, today
//! just Unix and Windows, and initial support for Redox.
//!
//! The centralization of platform-specific code in this module is
//! enforced by the "platform abstraction layer" tidy script in
//! `tools/tidy/src/pal.rs`.
//!
//! This module is closely related to the platform-independent system
//! integration code in `std::sys_common`. See that module's
//! documentation for details.
//!
//! In the future it would be desirable for the independent
//! implementations of this module to be extracted to their own crates
//! that `std` can link to, thus enabling their implementation
//! out-of-tree via crate replacement. Though due to the complex
//! inter-dependencies within `std` that will be a challenging goal to
//! achieve.

#![allow(missing_debug_implementations)]

pub mod common;

cfg_if::cfg_if! {
    if #[cfg(unix)] {
        mod unix;
        pub use self::unix::*;
    } else if #[cfg(windows)] {
        mod windows;
        pub use self::windows::*;
    } else if #[cfg(target_os = "solid_asp3")] {
        mod solid;
        pub use self::solid::*;
    } else if #[cfg(target_os = "hermit")] {
        mod hermit;
        pub use self::hermit::*;
    } else if #[cfg(target_os = "wasi")] {
        mod wasi;
        pub use self::wasi::*;
    } else if #[cfg(target_family = "wasm")] {
        mod wasm;
        pub use self::wasm::*;
    } else if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        mod sgx;
        pub use self::sgx::*;
    } else {
        mod unsupported;
        pub use self::unsupported::*;
    }
}

// Import essential modules from platforms used in `std::os` when documenting.
//
// Note that on some platforms those modules don't compile
// (missing things in `libc` which is empty), so they are not included in `std::os` and can be
// omitted here as well.

#[cfg(doc)]
#[cfg(not(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    all(target_vendor = "fortanix", target_env = "sgx")
)))]
cfg_if::cfg_if! {
    if #[cfg(not(windows))] {
        // On non-Windows platforms (aka linux/osx/etc) pull in a "minimal"
        // amount of windows goop which ends up compiling

        #[macro_use]
        #[path = "windows/compat.rs"]
        pub mod compat;

        #[path = "windows/c.rs"]
        pub mod c;
    }
}

#[doc(hidden)]
#[unstable(feature = "thread_local_internals", reason = "should not be necessary", issue = "none")]
#[macro_export]
#[allow_internal_unstable(thread_local_internals, cfg_target_thread_local, thread_local)]
#[allow_internal_unsafe]
macro_rules! __thread_local_inner {
    // used to generate the `LocalKey` value for const-initialized thread locals
    (@key $t:ty, const $init:expr) => {{
        #[cfg_attr(not(windows), inline)] // see comments below
        #[deny(unsafe_op_in_unsafe_fn)]
        unsafe fn __getit(
            _init: $crate::option::Option<&mut $crate::option::Option<$t>>,
        ) -> $crate::option::Option<&'static $t> {
            const INIT_EXPR: $t = $init;

            // wasm without atomics maps directly to `static mut`, and dtors
            // aren't implemented because thread dtors aren't really a thing
            // on wasm right now
            //
            // FIXME(#84224) this should come after the `target_thread_local`
            // block.
            #[cfg(all(target_family = "wasm", not(target_feature = "atomics")))]
            {
                static mut VAL: $t = INIT_EXPR;
                unsafe { $crate::option::Option::Some(&VAL) }
            }

            // If the platform has support for `#[thread_local]`, use it.
            #[cfg(all(
                target_thread_local,
                not(all(target_family = "wasm", not(target_feature = "atomics"))),
            ))]
            {
                #[thread_local]
                static mut VAL: $t = INIT_EXPR;

                // If a dtor isn't needed we can do something "very raw" and
                // just get going.
                if !$crate::mem::needs_drop::<$t>() {
                    unsafe {
                        return $crate::option::Option::Some(&VAL)
                    }
                }

                // 0 == dtor not registered
                // 1 == dtor registered, dtor not run
                // 2 == dtor registered and is running or has run
                #[thread_local]
                static mut STATE: $crate::primitive::u8 = 0;

                unsafe extern "C" fn destroy(ptr: *mut $crate::primitive::u8) {
                    let ptr = ptr as *mut $t;

                    unsafe {
                        $crate::debug_assert_eq!(STATE, 1);
                        STATE = 2;
                        $crate::ptr::drop_in_place(ptr);
                    }
                }

                unsafe {
                    match STATE {
                        // 0 == we haven't registered a destructor, so do
                        //   so now.
                        0 => {
                            $crate::thread::__FastLocalKeyInner::<$t>::register_dtor(
                                $crate::ptr::addr_of_mut!(VAL) as *mut $crate::primitive::u8,
                                destroy,
                            );
                            STATE = 1;
                            $crate::option::Option::Some(&VAL)
                        }
                        // 1 == the destructor is registered and the value
                        //   is valid, so return the pointer.
                        1 => $crate::option::Option::Some(&VAL),
                        // otherwise the destructor has already run, so we
                        // can't give access.
                        _ => $crate::option::Option::None,
                    }
                }
            }

            // On platforms without `#[thread_local]` we fall back to the
            // same implementation as below for os thread locals.
            #[cfg(all(
                not(target_thread_local),
                not(all(target_family = "wasm", not(target_feature = "atomics"))),
            ))]
            {
                #[inline]
                const fn __init() -> $t { INIT_EXPR }
                static __KEY: $crate::thread::__OsLocalKeyInner<$t> =
                    $crate::thread::__OsLocalKeyInner::new();
                #[allow(unused_unsafe)]
                unsafe {
                    __KEY.get(move || {
                        if let $crate::option::Option::Some(init) = _init {
                            if let $crate::option::Option::Some(value) = init.take() {
                                return value;
                            } else if $crate::cfg!(debug_assertions) {
                                $crate::unreachable!("missing initial value");
                            }
                        }
                        __init()
                    })
                }
            }
        }

        unsafe {
            $crate::thread::LocalKey::new(__getit)
        }
    }};

    // used to generate the `LocalKey` value for `thread_local!`
    (@key $t:ty, $init:expr) => {
        {
            #[inline]
            fn __init() -> $t { $init }

            // When reading this function you might ask "why is this inlined
            // everywhere other than Windows?", and that's a very reasonable
            // question to ask. The short story is that it segfaults rustc if
            // this function is inlined. The longer story is that Windows looks
            // to not support `extern` references to thread locals across DLL
            // boundaries. This appears to at least not be supported in the ABI
            // that LLVM implements.
            //
            // Because of this we never inline on Windows, but we do inline on
            // other platforms (where external references to thread locals
            // across DLLs are supported). A better fix for this would be to
            // inline this function on Windows, but only for "statically linked"
            // components. For example if two separately compiled rlibs end up
            // getting linked into a DLL then it's fine to inline this function
            // across that boundary. It's only not fine to inline this function
            // across a DLL boundary. Unfortunately rustc doesn't currently
            // have this sort of logic available in an attribute, and it's not
            // clear that rustc is even equipped to answer this (it's more of a
            // Cargo question kinda). This means that, unfortunately, Windows
            // gets the pessimistic path for now where it's never inlined.
            //
            // The issue of "should enable on Windows sometimes" is #84933
            #[cfg_attr(not(windows), inline)]
            unsafe fn __getit(
                init: $crate::option::Option<&mut $crate::option::Option<$t>>,
            ) -> $crate::option::Option<&'static $t> {
                #[cfg(all(target_family = "wasm", not(target_feature = "atomics")))]
                static __KEY: $crate::thread::__StaticLocalKeyInner<$t> =
                    $crate::thread::__StaticLocalKeyInner::new();

                #[thread_local]
                #[cfg(all(
                    target_thread_local,
                    not(all(target_family = "wasm", not(target_feature = "atomics"))),
                ))]
                static __KEY: $crate::thread::__FastLocalKeyInner<$t> =
                    $crate::thread::__FastLocalKeyInner::new();

                #[cfg(all(
                    not(target_thread_local),
                    not(all(target_family = "wasm", not(target_feature = "atomics"))),
                ))]
                static __KEY: $crate::thread::__OsLocalKeyInner<$t> =
                    $crate::thread::__OsLocalKeyInner::new();

                // FIXME: remove the #[allow(...)] marker when macros don't
                // raise warning for missing/extraneous unsafe blocks anymore.
                // See https://github.com/rust-lang/rust/issues/74838.
                #[allow(unused_unsafe)]
                unsafe {
                    __KEY.get(move || {
                        if let $crate::option::Option::Some(init) = init {
                            if let $crate::option::Option::Some(value) = init.take() {
                                return value;
                            } else if $crate::cfg!(debug_assertions) {
                                $crate::unreachable!("missing default value");
                            }
                        }
                        __init()
                    })
                }
            }

            unsafe {
                $crate::thread::LocalKey::new(__getit)
            }
        }
    };
    ($(#[$attr:meta])* $vis:vis $name:ident, $t:ty, $($init:tt)*) => {
        $(#[$attr])* $vis const $name: $crate::thread::LocalKey<$t> =
            $crate::__thread_local_inner!(@key $t, $($init)*);
    }
}
