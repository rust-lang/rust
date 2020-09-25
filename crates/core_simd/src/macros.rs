/// Provides implementations of `From<$a> for $b` and `From<$b> for $a` that transmutes the value.
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

/// Provides implementations of `From<$generic> for core::arch::{x86, x86_64}::$intel` and
/// vice-versa that transmutes the value.
macro_rules! from_transmute_x86 {
    { unsafe $generic:ty => $intel:ident } => {
        #[cfg(target_arch = "x86")]
        from_transmute! { unsafe $generic => core::arch::x86::$intel }

        #[cfg(target_arch = "x86_64")]
        from_transmute! { unsafe $generic => core::arch::x86_64::$intel }
    }
}

/// Calls a the macro `$mac` with the provided `$args` followed by `$repeat` repeated the specified
/// number of times.
macro_rules! call_repeat {
    { 1 => $mac:path [$($repeat:tt)*] $($args:tt)* } => {
        $mac! {
            $($args)*
            $($repeat)*
        }
    };
    { 2 => $mac:path [$($repeat:tt)*] $($args:tt)* } => {
        $mac! {
            $($args)*
            $($repeat)* $($repeat)*
        }
    };
    { 4 => $mac:path [$($repeat:tt)*] $($args:tt)* } => {
        $mac! {
            $($args)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)*
        }
    };
    { 8 => $mac:path [$($repeat:tt)*] $($args:tt)* } => {
        $mac! {
            $($args)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
        }
    };
    { 16 => $mac:path [$($repeat:tt)*] $($args:tt)* } => {
        $mac! {
            $($args)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
        }
    };
    { 32 => $mac:path [$($repeat:tt)*] $($args:tt)* } => {
        $mac! {
            $($args)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
        }
    };
    { 64 => $mac:path [$($repeat:tt)*] $($args:tt)* } => {
        $mac! {
            $($args)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
            $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)* $($repeat)*
        }
    };
}

/// Calls the macro `$mac` with the specified `$args` followed by the specified number of unique
/// identifiers.
macro_rules! call_counting_args {
    { 1 => $mac:path => $($args:tt)* } => {
        $mac! {
            $($args)*
            value
        }
    };
    { 2 => $mac:path => $($args:tt)* } => {
        $mac! {
            $($args)*
            v0 v1
        }
    };
    { 4 => $mac:path => $($args:tt)* } => {
        $mac! {
            $($args)*
            v0 v1 v2 v3
        }
    };
    { 8 => $mac:path => $($args:tt)* } => {
        $mac! {
            $($args)*
            v0 v1 v2 v3 v4 v5 v6 v7
        }
    };
    { 16 => $mac:path => $($args:tt)* } => {
        $mac! {
            $($args)*
            v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15
        }
    };
    { 32 => $mac:path => $($args:tt)* } => {
        $mac! {
            $($args)*
            v0  v1  v2  v3  v4  v5  v6  v7  v8  v9  v10 v11 v12 v13 v14 v15
            v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31
        }
    };
    { 64 => $mac:path => $($args:tt)* } => {
        $mac! {
            $($args)*
            v0  v1  v2  v3  v4  v5  v6  v7  v8  v9  v10 v11 v12 v13 v14 v15
            v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31
            v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47
            v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63
        }
    };
}

/// Calls the macro `$mac` with the specified `$args` followed by counting values from 0 to the
/// specified value.
macro_rules! call_counting_values {
    { 1 => $mac:path => $($args:tt)* } => {
        $mac! {
            $($args)*
            0
        }
    };
    { 2 => $mac:path => $($args:tt)* } => {
        $mac! {
            $($args)*
            0 1
        }
    };
    { 4 => $mac:path => $($args:tt)* } => {
        $mac! {
            $($args)*
            0 1 2 3
        }
    };
    { 8 => $mac:path => $($args:tt)* } => {
        $mac! {
            $($args)*
            0 1 2 3 4 5 6 7
        }
    };
    { 16 => $mac:path => $($args:tt)* } => {
        $mac! {
            $($args)*
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
        }
    };
    { 32 => $mac:path => $($args:tt)* } => {
        $mac! {
            $($args)*
            0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
            16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
        }
    };
    { 64 => $mac:path => $($args:tt)* } => {
        $mac! {
            $($args)*
            0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
            16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
            32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
            48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
        }
    };
}

/// Implements common traits on the specified vector `$name`, holding multiple `$lanes` of `$type`.
macro_rules! base_vector_traits {
    { $name:path => [$type:ty; $lanes:literal] } => {
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
    }
}

/// Defines a vector `$name` containing multiple `$lanes` of `$type`.
macro_rules! define_vector {
    { $(#[$attr:meta])* struct $name:ident([$type:ty; $lanes:tt]); } => {
        call_repeat! { $lanes => define_vector [$type] def $(#[$attr])* | $name | }

        impl $name {
            call_repeat! { $lanes => define_vector [$type] splat $type | }
            call_counting_args! { $lanes => define_vector => new $type | }
        }

        base_vector_traits! { $name => [$type; $lanes] }
    };
    { def $(#[$attr:meta])* | $name:ident | $($itype:ty)* } => {
        $(#[$attr])*
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Default, PartialEq, PartialOrd)]
        #[repr(simd)]
        pub struct $name($($itype),*);
    };
    { splat $type:ty | $($itype:ty)* } => {
        /// Construct a vector by setting all lanes to the given value.
        #[inline]
        pub const fn splat(value: $type) -> Self {
            Self($(value as $itype),*)
        }
    };
    { new $type:ty | $($var:ident)* } => {
        /// Construct a vector by setting each lane to the given values.
        #[allow(clippy::too_many_arguments)]
        #[inline]
        pub const fn new($($var: $type),*) -> Self {
            Self($($var),*)
        }
    }
}

/// Defines a mask vector `$name` containing multiple `$lanes` of `$type`, represented by the
/// underlying type `$impl_type`.
macro_rules! define_mask_vector {
    { $(#[$attr:meta])* struct $name:ident([$impl_type:ty as $type:ty; $lanes:tt]); } => {
        call_repeat! { $lanes => define_mask_vector [$impl_type] def $(#[$attr])* | $name | }

        impl $name {
            call_repeat! { $lanes => define_mask_vector [$impl_type] splat $type | }
            call_counting_args! { $lanes => define_mask_vector => new $type | }
        }

        base_vector_traits! { $name => [$type; $lanes] }
    };
    { def $(#[$attr:meta])* | $name:ident | $($itype:ty)* } => {
        $(#[$attr])*
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Default, PartialEq, PartialOrd, Eq, Ord)]
        #[repr(simd)]
        pub struct $name($($itype),*);
    };
    { splat $type:ty | $($itype:ty)* } => {
        /// Construct a vector by setting all lanes to the given value.
        #[inline]
        pub const fn splat(value: $type) -> Self {
            Self($(value.0 as $itype),*)
        }
    };
    { new $type:ty | $($var:ident)* } => {
        /// Construct a vector by setting each lane to the given values.
        #[allow(clippy::too_many_arguments)]
        #[inline]
        pub const fn new($($var: $type),*) -> Self {
            Self($($var.0),*)
        }
    }
}
