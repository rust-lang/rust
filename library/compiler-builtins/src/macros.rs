macro_rules! intrinsics {
    () => ();

    // Anything which has a `not(feature = "c")` we'll generate a shim function
    // which calls out to the C function if the `c` feature is enabled.
    // Otherwise if the `c` feature isn't enabled then we'll just have a normal
    // intrinsic.
    (
        #[use_c_shim_if($($cfg_clause:tt)*)]
        $(#[$($attr:tt)*])*
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) -> $ret:ty {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (

        #[cfg(all(feature = "c", $($cfg_clause)*))]
        pub extern $abi fn $name( $($argname: $ty),* ) -> $ret {
            extern $abi {
                fn $name($($argname: $ty),*) -> $ret;
            }
            unsafe {
                $name($($argname),*)
            }
        }

        #[cfg(not(all(feature = "c", $($cfg_clause)*)))]
        intrinsics! {
            $(#[$($attr)*])*
            pub extern $abi fn $name( $($argname: $ty),* ) -> $ret {
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
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) -> $ret:ty {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        #[cfg(target_arch = "arm")]
        intrinsics! {
            $(#[$($attr)*])*
            pub extern "aapcs" fn $name( $($argname: $ty),* ) -> $ret {
                $($body)*
            }
        }

        #[cfg(not(target_arch = "arm"))]
        intrinsics! {
            $(#[$($attr)*])*
            pub extern $abi fn $name( $($argname: $ty),* ) -> $ret {
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
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) -> $ret:ty {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        #[cfg(all(windows, target_pointer_width = "64"))]
        intrinsics! {
            $(#[$($attr)*])*
            pub extern "unadjusted" fn $name( $($argname: $ty),* ) -> $ret {
                $($body)*
            }
        }

        #[cfg(not(all(windows, target_pointer_width = "64")))]
        intrinsics! {
            $(#[$($attr)*])*
            pub extern $abi fn $name( $($argname: $ty),* ) -> $ret {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );

    // Another attribute we recognize is an "abi hack" for win64 to get the 128
    // bit calling convention correct.
    (
        #[win64_128bit_abi_hack]
        $(#[$($attr:tt)*])*
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) -> $ret:ty {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        #[cfg(all(windows, target_pointer_width = "64"))]
        $(#[$($attr)*])*
        pub extern $abi fn $name( $($argname: $ty),* ) -> $ret {
            $($body)*
        }

        #[cfg(all(windows, target_pointer_width = "64"))]
        pub mod $name {

            intrinsics! {
                pub extern $abi fn $name( $($argname: $ty),* )
                    -> ::macros::win64_abi_hack::U64x2
                {
                    let e: $ret = super::$name($($argname),*);
                    ::macros::win64_abi_hack::U64x2::from(e)
                }
            }
        }

        #[cfg(not(all(windows, target_pointer_width = "64")))]
        intrinsics! {
            $(#[$($attr)*])*
            pub extern $abi fn $name( $($argname: $ty),* ) -> $ret {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );

    // A bunch of intrinsics on ARM are aliased in the standard compiler-rt
    // build under `__aeabi_*` aliases, and LLVM will call these instead of the
    // original function. Handle that here
    (
        #[arm_aeabi_alias = $alias:ident]
        $(#[$($attr:tt)*])*
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) -> $ret:ty {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        #[cfg(target_arch = "arm")]
        $(#[$($attr)*])*
        pub extern $abi fn $name( $($argname: $ty),* ) -> $ret {
            $($body)*
        }

        #[cfg(target_arch = "arm")]
        pub mod $name {
            intrinsics! {
                pub extern "aapcs" fn $alias( $($argname: $ty),* ) -> $ret {
                    super::$name($($argname),*)
                }

                pub extern $abi fn $name( $($argname: $ty),* ) -> $ret {
                    super::$name($($argname),*)
                }
            }
        }

        #[cfg(not(target_arch = "arm"))]
        intrinsics! {
            $(#[$($attr)*])*
            pub extern $abi fn $name( $($argname: $ty),* ) -> $ret {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );

    (
        $(#[$($attr:tt)*])*
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) -> $ret:ty {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        $(#[$($attr)*])*
        #[cfg_attr(not(test), no_mangle)]
        pub extern $abi fn $name( $($argname: $ty),* ) -> $ret {
            $($body)*
        }

        intrinsics!($($rest)*);
    );
}

// Hack for LLVM expectations for ABI on windows
#[cfg(all(windows, target_pointer_width="64"))]
pub mod win64_abi_hack {
    #[repr(simd)]
    pub struct U64x2(u64, u64);

    impl From<i128> for U64x2 {
        fn from(i: i128) -> U64x2 {
            use int::LargeInt;
            let j = i as u128;
            U64x2(j.low(), j.high())
        }
    }

    impl From<u128> for U64x2 {
        fn from(i: u128) -> U64x2 {
            use int::LargeInt;
            U64x2(i.low(), i.high())
        }
    }
}
