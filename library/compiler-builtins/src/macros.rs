macro_rules! intrinsics {
    () => ();

    // Anything which has a `not(feature = "c")` we'll generate a shim function
    // which calls out to the C function if the `c` feature is enabled.
    // Otherwise if the `c` feature isn't enabled then we'll just have a normal
    // intrinsic.
    (
        #[cfg(not(all(feature = "c", $($cfg_clause:tt)*)))]
        $(#[$attr:meta])*
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) -> $ret:ty {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (

        #[cfg(all(feature = "c", not($($cfg_clause)*)))]
        $(#[$attr])*
        pub extern $abi fn $name( $($argname: $ty),* ) -> $ret {
            extern $abi {
                fn $name($($argname: $ty),*) -> $ret;
            }
            unsafe {
                $name($($argname),*)
            }
        }

        #[cfg(not(all(feature = "c", not($($cfg_clause)*))))]
        intrinsics! {
            $(#[$attr])*
            pub extern $abi fn $name( $($argname: $ty),* ) -> $ret {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );

    // We recognize the `#[aapcs_only_on_arm]` attribute here and generate the
    // same intrinsic but force it to have the `"aapcs"` calling convention on
    // ARM and `"C"` elsewhere.
    (
        #[aapcs_on_arm]
        $(#[$attr:meta])*
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) -> $ret:ty {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        #[cfg(target_arch = "arm")]
        intrinsics! {
            $(#[$attr])*
            pub extern "aapcs" fn $name( $($argname: $ty),* ) -> $ret {
                $($body)*
            }
        }

        #[cfg(not(target_arch = "arm"))]
        intrinsics! {
            $(#[$attr])*
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
        $(#[$attr:meta])*
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) -> $ret:ty {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        #[cfg(all(windows, target_pointer_width = "64"))]
        intrinsics! {
            $(#[$attr])*
            pub extern "unadjusted" fn $name( $($argname: $ty),* ) -> $ret {
                $($body)*
            }
        }

        #[cfg(not(all(windows, target_pointer_width = "64")))]
        intrinsics! {
            $(#[$attr])*
            pub extern $abi fn $name( $($argname: $ty),* ) -> $ret {
                $($body)*
            }
        }

        intrinsics!($($rest)*);
    );

    (
        $(#[$attr:meta])*
        pub extern $abi:tt fn $name:ident( $($argname:ident:  $ty:ty),* ) -> $ret:ty {
            $($body:tt)*
        }

        $($rest:tt)*
    ) => (
        $(#[$attr])*
        #[cfg_attr(not(test), no_mangle)]
        pub extern $abi fn $name( $($argname: $ty),* ) -> $ret {
            $($body)*
        }

        intrinsics!($($rest)*);
    );
}
