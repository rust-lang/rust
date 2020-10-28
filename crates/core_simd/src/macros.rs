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

/// Implements common traits on the specified vector `$name`, holding multiple `$lanes` of `$type`.
macro_rules! base_vector_traits {
    { $name:path => [$type:ty; $lanes:literal] } => {
        impl Copy for $name {}

        impl Clone for $name {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl Default for $name {
            #[inline]
            fn default() -> Self {
                Self::splat(<$type>::default())
            }
        }

        impl PartialEq for $name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                AsRef::<[$type]>::as_ref(self) == AsRef::<[$type]>::as_ref(other)
            }
        }

        impl PartialOrd for $name {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                AsRef::<[$type]>::as_ref(self).partial_cmp(AsRef::<[$type]>::as_ref(other))
            }
        }

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

/// Implements additional integer traits (Eq, Ord, Hash) on the specified vector `$name`, holding multiple `$lanes` of `$type`.
macro_rules! integer_vector_traits {
    { $name:path => [$type:ty; $lanes:literal] } => {
        impl Eq for $name {}

        impl Ord for $name {
            #[inline]
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                AsRef::<[$type]>::as_ref(self).cmp(AsRef::<[$type]>::as_ref(other))
            }
        }

        impl core::hash::Hash for $name {
            #[inline]
            fn hash<H>(&self, state: &mut H)
            where
                H: core::hash::Hasher
            {
                AsRef::<[$type]>::as_ref(self).hash(state)
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

/// Implements inherent methods for a float vector `$name` containing multiple
/// `$lanes` of float `$type`, which uses `$bits_ty` as its binary
/// representation. Called from `define_float_vector!`.
macro_rules! impl_float_vector {
    { $name:path => [$type:ty; $lanes:literal]; bits $bits_ty:ty; } => {
        impl $name {
            /// Raw transmutation to an unsigned integer vector type with the
            /// same size and number of lanes.
            #[inline]
            pub fn to_bits(self) -> $bits_ty {
                unsafe { core::mem::transmute(self) }
            }

            /// Raw transmutation from an unsigned integer vector type with the
            /// same size and number of lanes.
            #[inline]
            pub fn from_bits(bits: $bits_ty) -> Self {
                unsafe { core::mem::transmute(bits) }
            }

            /// Produces a vector where every lane has the absolute value of the
            /// equivalently-indexed lane in `self`.
            #[inline]
            pub fn abs(self) -> Self {
                let no_sign = <$bits_ty>::splat(!0 >> 1);
                Self::from_bits(self.to_bits() & no_sign)
            }
        }
    };
}

/// Defines a float vector `$name` containing multiple `$lanes` of float
/// `$type`, which uses `$bits_ty` as its binary representation.
macro_rules! define_float_vector {
    { $(#[$attr:meta])* struct $name:ident([$type:ty; $lanes:tt]); bits $bits_ty:ty; } => {
        define_vector! {
            $(#[$attr])*
            struct $name([$type; $lanes]);
        }

        impl_float_vector! { $name => [$type; $lanes]; bits $bits_ty; }
    }
}

/// Defines an integer vector `$name` containing multiple `$lanes` of integer `$type`.
macro_rules! define_integer_vector {
    { $(#[$attr:meta])* struct $name:ident([$type:ty; $lanes:tt]); } => {
        define_vector! {
            $(#[$attr])*
            struct $name([$type; $lanes]);
        }

        integer_vector_traits! { $name => [$type; $lanes] }
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
            call_counting_args! { $lanes => define_mask_vector => new_from_bool $type | }

            /// Tests the value of the specified lane.
            ///
            /// # Panics
            /// Panics if `lane` is greater than or equal to the number of lanes in the vector.
            #[inline]
            pub fn test(&self, lane: usize) -> bool {
                self[lane].test()
            }

            /// Sets the value of the specified lane.
            ///
            /// # Panics
            /// Panics if `lane` is greater than or equal to the number of lanes in the vector.
            #[inline]
            pub fn set(&mut self, lane: usize, value: bool) {
                self[lane] = value.into();
            }
        }

        base_vector_traits! { $name => [$type; $lanes] }
        integer_vector_traits! { $name => [$type; $lanes] }
    };
    { def $(#[$attr:meta])* | $name:ident | $($itype:ty)* } => {
        $(#[$attr])*
        #[allow(non_camel_case_types)]
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
    };
    { new_from_bool $type:ty | $($var:ident)* } => {
        /// Used internally (since we can't use the Into trait in `const fn`s)
        #[allow(clippy::too_many_arguments)]
        #[allow(unused)]
        #[inline]
        pub(crate) const fn new_from_bool($($var: bool),*) -> Self {
            Self($(<$type>::new($var).0),*)
        }
    }
}
