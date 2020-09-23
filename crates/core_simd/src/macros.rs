macro_rules! from_transmute {
    { unsafe $a:ty => $b:ty } => {
        from_transmute!{ @impl $a => $b }
        from_transmute!{ @impl $b => $a }
    };
    { @impl $from:ty => $to:ty } => {
        impl core::convert::From<$from> for $to {
            #[inline]
            fn from(value: $from) -> $to {
                unsafe { core::mem::transmute(value) }
            }
        }
    };
}

macro_rules! from_transmute_x86 {
    { unsafe $generic:ty => $intel:ident } => {
        #[cfg(target_arch = "x86")]
        from_transmute! { unsafe $generic => core::arch::x86::$intel }

        #[cfg(target_arch = "x86_64")]
        from_transmute! { unsafe $generic => core::arch::x86_64::$intel }
    }
}

macro_rules! define_type {
    { $(#[$attr:meta])* struct $name:ident([$type:ty; $lanes:tt]); } => {
        define_type! { @impl $(#[$attr])* | $name [$type; $lanes] }

        // array references
        impl AsRef<[$type; $lanes]> for $name {
            #[inline]
            fn as_ref(&self) -> &[$type; $lanes] {
                unsafe { &*(self as *const _ as *const _) }
            }
        }

        impl AsMut<[$type; $lanes]> for $name {
            #[inline]
            fn as_mut(&mut self) -> &mut [$type; $lanes] {
                unsafe { &mut *(self as *mut _ as *mut _) }
            }
        }

        // slice references
        impl AsRef<[$type]> for $name {
            #[inline]
            fn as_ref(&self) -> &[$type] {
                AsRef::<[$type; $lanes]>::as_ref(self)
            }
        }

        impl AsMut<[$type]> for $name {
            #[inline]
            fn as_mut(&mut self) -> &mut [$type] {
                AsMut::<[$type; $lanes]>::as_mut(self)
            }
        }

        // vector/array conversion
        from_transmute! { unsafe $name => [$type; $lanes] }

        // splat
        impl From<$type> for $name {
            #[inline]
            fn from(value: $type) -> Self {
                Self::splat(value)
            }
        }
    };
    { @impl $(#[$attr:meta])* | $name:ident [$type:ty; 1] } => {
        define_type! { @def $(#[$attr])* | $name | $type | $type, | v0, }
    };
    { @impl $(#[$attr:meta])* | $name:ident [$type:ty; 2] } => {
        define_type! { @def $(#[$attr])* | $name | $type | $type, $type, | v0, v1, }
    };
    { @impl $(#[$attr:meta])* | $name:ident [$type:ty; 4] } => {
        define_type! { @def $(#[$attr])* | $name | $type |
            $type, $type, $type, $type, |
            v0, v1, v2, v3,
        }
    };
    { @impl $(#[$attr:meta])* | $name:ident [$type:ty; 8] } => {
        define_type! { @def $(#[$attr])* | $name | $type |
            $type, $type, $type, $type, $type, $type, $type, $type, |
            v0, v1, v2, v3, v4, v5, v6, v7,
        }
    };
    { @impl $(#[$attr:meta])* | $name:ident [$type:ty; 16] } => {
        define_type! { @def $(#[$attr])* | $name | $type |
            $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, |
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
        }
    };
    { @impl $(#[$attr:meta])* | $name:ident [$type:ty; 32] } => {
        define_type! { @def $(#[$attr])* | $name | $type |
            $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type,
            $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, |
            v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,  v8,  v9,  v10, v11, v12, v13, v14, v15,
            v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31,
        }
    };
    { @impl $(#[$attr:meta])* | $name:ident [$type:ty; 64] } => {
        define_type! { @def $(#[$attr])* | $name | $type |
            $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type,
            $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type,
            $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type,
            $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, $type, |
            v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,  v8,  v9,  v10, v11, v12, v13, v14, v15,
            v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31,
            v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47,
            v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63,
        }
    };
    { @def $(#[$attr:meta])* | $name:ident | $type:ty | $($itype:ty,)* | $($ivar:ident,)* } => {
        $(#[$attr])*
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
        #[repr(simd)]
        pub struct $name($($itype),*);

        impl $name {
            /// Construct a vector by setting all lanes to the given value.
            #[inline]
            pub const fn splat(value: $type) -> Self {
                Self($(value as $itype),*)
            }

            /// Construct a vector by setting each lane to the given values.
            #[allow(clippy::too_many_arguments)]
            #[inline]
            pub const fn new($($ivar: $itype),*) -> Self {
                Self($($ivar),*)
            }
        }
    }
}
