//! Macros shared throughout the compiler-builtins implementation

/// The "main macro" used for defining intrinsics.
///
/// The compiler-builtins library is super platform-specific with tons of crazy
/// little tweaks for various platforms. As a result it *could* involve a lot of
/// #[cfg] and macro soup, but the intention is that this macro alleviates a lot
/// of that complexity. Ideally this macro has all the weird ABI things
/// platforms need and elsewhere in this library it just looks like normal Rust
/// code.
///
/// All intrinsics functions are marked with #[linkage = "weak"] when
/// `not(windows) and not(target_vendor = "apple")`.
/// `weak` linkage attribute is used so that these functions can be replaced
/// by another implementation at link time. This is particularly useful for mixed
/// Rust/C++ binaries that want to use the C++ intrinsics, otherwise linking against
/// the Rust stdlib will replace those from the compiler-rt library.
///
/// This macro is structured to be invoked with a bunch of functions that looks
/// like:
/// ```ignore
///     intrinsics! {
///         pub extern "C" fn foo(a: i32) -> u32 {
///             // ...
///         }
///
///         #[nonstandard_attribute]
///         pub extern "C" fn bar(a: i32) -> u32 {
///             // ...
///         }
///     }
/// ```
///
/// Each function is defined in a manner that looks like a normal Rust function.
/// The macro then accepts a few nonstandard attributes that can decorate
/// various functions. Each of the attributes is documented below with what it
/// can do, and each of them slightly tweaks how further expansion happens.
///
/// A quick overview of attributes supported right now are:
///
/// * `maybe_use_optimized_c_shim` - indicates that the Rust implementation is
///   ignored if an optimized C version was compiled.
/// * `aapcs_on_arm` - forces the ABI of the function to be `"aapcs"` on ARM and
///   the specified ABI everywhere else.
/// * `unadjusted_on_win64` - like `aapcs_on_arm` this switches to the
///   `"unadjusted"` abi on Win64 and the specified abi elsewhere.
/// * `arm_aeabi_alias` - handles the "aliasing" of various intrinsics on ARM
///   their otherwise typical names to other prefixed ones.
/// * `ppc_alias` - changes the name of the symbol on PowerPC platforms without
///   changing any other behavior. This is mostly for `f128`, which is `tf` on
///   most platforms but `kf` on PowerPC.
macro_rules! intrinsics {
    () => ();

    // Support cfg_attr:
    (
        #[cfg_attr($e:meta, $($attr:tt)*)]
        $(#[$($attrs:tt)*])*
        pub extern $abi:tt fn $name:ident( $($argname:ident: $ty:ty),* ) $(-> $ret:ty)? {
            $($body:tt)*
        }
        $($rest:tt)*
    ) => (
        #[cfg($e)]
        intrinsics! {
            #[$($attr)*]
            $(#[$($attrs)*])*
            pub extern $abi fn $name($($argname: $ty),*) $(-> $ret)? {
                $($body)*
            }
        }

        #[cfg(not($e))]
        intrinsics! {
            $(#[$($attrs)*])*
            pub extern $abi fn $name($($argname: $ty),*) $(-> $ret)? {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );
    // Same as above but for unsafe.
    (
        #[cfg_attr($e:meta, $($attr:tt)*)]
        $(#[$($attrs:tt)*])*
        pub unsafe extern $abi:tt fn $name:ident( $($argname:ident: $ty:ty),* ) $(-> $ret:ty)? {
            $($body:tt)*
        }
        $($rest:tt)*
    ) => (
        #[cfg($e)]
        intrinsics! {
            #[$($attr)*]
            $(#[$($attrs)*])*
            pub unsafe extern $abi fn $name($($argname: $ty),*) $(-> $ret)? {
                $($body)*
            }
        }

        #[cfg(not($e))]
        intrinsics! {
            $(#[$($attrs)*])*
            pub unsafe extern $abi fn $name($($argname: $ty),*) $(-> $ret)? {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );

    // Right now there's a bunch of architecture-optimized intrinsics in the
    // stock compiler-rt implementation. Not all of these have been ported over
    // to Rust yet so when the `c` feature of this crate is enabled we fall back
    // to the architecture-specific versions which should be more optimized. The
    // purpose of this macro is to easily allow specifying this.
    //
    // The `#[maybe_use_optimized_c_shim]` attribute indicates that this
    // intrinsic may have an optimized C version. In these situations the build
    // script, if the C code is enabled and compiled, will emit a cfg directive
    // to get passed to rustc for our compilation. If that cfg is set we skip
    // the Rust implementation, but if the attribute is not enabled then we
    // compile in the Rust implementation.
    (
        #[maybe_use_optimized_c_shim]
        $(#[$($attr:tt)*])*
        pub $(unsafe $(@ $empty:tt)? )? extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) $(-> $ret:ty)? {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        #[cfg($name = "optimized-c")]
        pub $(unsafe $($empty)? )? extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
            unsafe extern $abi {
                fn $name($($argname: $ty),*) $(-> $ret)?;
            }
            unsafe {
                $name($($argname),*)
            }
        }

        #[cfg(not($name = "optimized-c"))]
        intrinsics! {
            $(#[$($attr)*])*
            pub $(unsafe $($empty)? )? extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );

    // We recognize the `#[aapcs_on_arm]` attribute here and generate the
    // same intrinsic but force it to have the `"aapcs"` calling convention on
    // ARM and `"C"` elsewhere.
    (
        #[aapcs_on_arm]
        $(#[$($attr:tt)*])*
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) $(-> $ret:ty)? {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        #[cfg(target_arch = "arm")]
        intrinsics! {
            $(#[$($attr)*])*
            pub extern "aapcs" fn $name( $($argname: $ty),* ) $(-> $ret)? {
                $($body)*
            }
        }

        #[cfg(not(target_arch = "arm"))]
        intrinsics! {
            $(#[$($attr)*])*
            pub extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );

    // Like aapcs above we recognize an attribute for the "unadjusted" abi on
    // win64 for some methods.
    (
        #[unadjusted_on_win64]
        $(#[$($attr:tt)*])*
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) $(-> $ret:ty)? {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        #[cfg(all(any(windows, target_os = "cygwin", all(target_os = "uefi", target_arch = "x86_64")), target_pointer_width = "64"))]
        intrinsics! {
            $(#[$($attr)*])*
            pub extern "unadjusted" fn $name( $($argname: $ty),* ) $(-> $ret)? {
                $($body)*
            }
        }

        #[cfg(not(all(any(windows, target_os = "cygwin", all(target_os = "uefi", target_arch = "x86_64")), target_pointer_width = "64")))]
        intrinsics! {
            $(#[$($attr)*])*
            pub extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );

    // `arm_aeabi_alias` would conflict with `f16_apple_{arg,ret}_abi` not handled here. Avoid macro ambiguity by combining in a
    // single `#[]`.
    (
        #[apple_f16_arg_abi]
        #[arm_aeabi_alias = $alias:ident]
        $($t:tt)*
    ) => {
        intrinsics! {
            #[apple_f16_arg_abi, arm_aeabi_alias = $alias]
            $($t)*
        }
    };
    (
        #[apple_f16_ret_abi]
        #[arm_aeabi_alias = $alias:ident]
        $($t:tt)*
    ) => {
        intrinsics! {
            #[apple_f16_ret_abi, arm_aeabi_alias = $alias]
            $($t)*
        }
    };

    // On x86 (32-bit and 64-bit) Apple platforms, `f16` is passed and returned like a `u16` unless
    // the builtin involves `f128`.
    (
        // `arm_aeabi_alias` would conflict if not handled here. Avoid macro ambiguity by combining
        // in a single `#[]`.
        #[apple_f16_arg_abi $(, arm_aeabi_alias = $alias:ident)?]
        $(#[$($attr:tt)*])*
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) $(-> $ret:ty)? {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        #[cfg(all(target_vendor = "apple", any(target_arch = "x86", target_arch = "x86_64")))]
        $(#[$($attr)*])*
        pub extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
            $($body)*
        }

        #[cfg(all(target_vendor = "apple", any(target_arch = "x86", target_arch = "x86_64"), not(feature = "mangled-names")))]
        mod $name {
            #[unsafe(no_mangle)]
            #[cfg_attr(not(any(all(windows, target_env = "gnu"), target_os = "cygwin")), linkage = "weak")]
            $(#[$($attr)*])*
            extern $abi fn $name( $($argname: u16),* ) $(-> $ret)? {
                super::$name($(f16::from_bits($argname)),*)
            }
        }

        #[cfg(not(all(target_vendor = "apple", any(target_arch = "x86", target_arch = "x86_64"))))]
        intrinsics! {
            $(#[arm_aeabi_alias = $alias])?
            $(#[$($attr)*])*
            pub extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );
    (
        #[apple_f16_ret_abi $(, arm_aeabi_alias = $alias:ident)?]
        $(#[$($attr:tt)*])*
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) $(-> $ret:ty)? {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        #[cfg(all(target_vendor = "apple", any(target_arch = "x86", target_arch = "x86_64")))]
        $(#[$($attr)*])*
        pub extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
            $($body)*
        }

        #[cfg(all(target_vendor = "apple", any(target_arch = "x86", target_arch = "x86_64"), not(feature = "mangled-names")))]
        mod $name {
            #[unsafe(no_mangle)]
            #[cfg_attr(not(any(all(windows, target_env = "gnu"), target_os = "cygwin")), linkage = "weak")]
            $(#[$($attr)*])*
            extern $abi fn $name( $($argname: $ty),* ) -> u16 {
                super::$name($($argname),*).to_bits()
            }
        }

        #[cfg(not(all(target_vendor = "apple", any(target_arch = "x86", target_arch = "x86_64"))))]
        intrinsics! {
            $(#[arm_aeabi_alias = $alias])?
            $(#[$($attr)*])*
            pub extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );

    // A bunch of intrinsics on ARM are aliased in the standard compiler-rt
    // build under `__aeabi_*` aliases, and LLVM will call these instead of the
    // original function. The aliasing here is used to generate these symbols in
    // the object file.
    (
        #[arm_aeabi_alias = $alias:ident]
        $(#[$($attr:tt)*])*
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) $(-> $ret:ty)? {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        #[cfg(target_arch = "arm")]
        $(#[$($attr)*])*
        pub extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
            $($body)*
        }

        #[cfg(all(target_arch = "arm", not(feature = "mangled-names")))]
        mod $name {
            #[unsafe(no_mangle)]
            #[cfg_attr(not(any(all(windows, target_env = "gnu"), target_os = "cygwin")), linkage = "weak")]
            $(#[$($attr)*])*
            extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
                super::$name($($argname),*)
            }
        }

        #[cfg(all(target_arch = "arm", not(feature = "mangled-names")))]
        mod $alias {
            #[unsafe(no_mangle)]
            #[cfg_attr(not(any(all(windows, target_env = "gnu"), target_os = "cygwin")), linkage = "weak")]
            $(#[$($attr)*])*
            extern "aapcs" fn $alias( $($argname: $ty),* ) $(-> $ret)? {
                super::$name($($argname),*)
            }
        }

        #[cfg(not(target_arch = "arm"))]
        intrinsics! {
            $(#[$($attr)*])*
            pub extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );

    // PowerPC usually uses `kf` rather than `tf` for `f128`. This is just an easy
    // way to add an alias on those targets.
    (
        #[ppc_alias = $alias:ident]
        $(#[$($attr:tt)*])*
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) $(-> $ret:ty)? {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        #[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
        intrinsics! {
            $(#[$($attr)*])*
            pub extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
                $($body)*
            }
        }

        #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
        intrinsics! {
            $(#[$($attr)*])*
            pub extern $abi fn $alias( $($argname: $ty),* ) $(-> $ret)? {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );

    // C mem* functions are only generated when the "mem" feature is enabled.
    (
        #[mem_builtin]
        $(#[$($attr:tt)*])*
        pub unsafe extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) $(-> $ret:ty)? {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        $(#[$($attr)*])*
        pub unsafe extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
            $($body)*
        }

        #[cfg(all(feature = "mem", not(feature = "mangled-names")))]
        mod $name {
            $(#[$($attr)*])*
            #[unsafe(no_mangle)]
            #[cfg_attr(not(any(all(windows, target_env = "gnu"), target_os = "cygwin")), linkage = "weak")]
            unsafe extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
                super::$name($($argname),*)
            }
        }

        intrinsics!($($rest)*);
    );

    // Naked functions are special: we can't generate wrappers for them since
    // they use a custom calling convention.
    (
        #[unsafe(naked)]
        $(#[$($attr:tt)*])*
        pub unsafe extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) $(-> $ret:ty)? {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        // `#[naked]` definitions are referenced by other places, so we can't use `cfg` like the others
        pub mod $name {
            #[unsafe(naked)]
            $(#[$($attr)*])*
            #[cfg_attr(not(feature = "mangled-names"), unsafe(no_mangle))]
            #[cfg_attr(not(any(all(windows, target_env = "gnu"), target_os = "cygwin")), linkage = "weak")]
            pub unsafe extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );

    // This is the final catch-all rule. At this point we generate an
    // intrinsic with a conditional `#[no_mangle]` directive to avoid
    // interfering with duplicate symbols and whatnot during testing.
    //
    // The implementation is placed in a separate module, to take advantage
    // of the fact that rustc partitions functions into code generation
    // units based on module they are defined in. As a result we will have
    // a separate object file for each intrinsic. For further details see
    // corresponding PR in rustc https://github.com/rust-lang/rust/pull/70846
    //
    // After the intrinsic is defined we just continue with the rest of the
    // input we were given.
    (
        $(#[$($attr:tt)*])*
        pub $(unsafe $(@ $empty:tt)?)? extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) $(-> $ret:ty)? {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        $(#[$($attr)*])*
        pub $(unsafe $($empty)?)? extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
            $($body)*
        }

        #[cfg(not(feature = "mangled-names"))]
        mod $name {
            $(#[$($attr)*])*
            #[unsafe(no_mangle)]
            #[cfg_attr(not(any(all(windows, target_env = "gnu"), target_os = "cygwin")), linkage = "weak")]
            $(unsafe $($empty)?)? extern $abi fn $name( $($argname: $ty),* ) $(-> $ret)? {
                // SAFETY: same preconditions.
                $(unsafe $($empty)?)? { super::$name($($argname),*) }
            }
        }

        intrinsics!($($rest)*);
    );
}
