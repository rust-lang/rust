macro_rules! intrinsics {
    () => ();
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
