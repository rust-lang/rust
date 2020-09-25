use core::marker::PhantomData;

use crate::vectors_isize::*;

macro_rules! define_pointer_vector {
    { $(#[$attr:meta])* $name:ident => $underlying:ty => $lanes:tt, $mut:ident } => {
        $(#[$attr])*
        #[allow(non_camel_case_types)]
        #[repr(C)]
        pub struct $name<T>($underlying, PhantomData<T>);

        impl<T> core::fmt::Debug for $name<T> {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                crate::fmt::format(self.as_ref(), f)
            }
        }
        impl<T> core::fmt::Pointer for $name<T> {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                crate::fmt::format_pointer(self.as_ref(), f)
            }
        }

        impl<T> Copy for $name<T> {}

        impl<T> Clone for $name<T> {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<T> core::cmp::PartialEq for $name<T> {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.0.eq(&other.0)
            }
        }

        impl<T> core::cmp::Eq for $name<T> {}

        impl<T> core::cmp::PartialOrd for $name<T> {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                self.0.partial_cmp(&other.0)
            }
        }

        impl<T> core::cmp::Ord for $name<T> {
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                self.0.cmp(&other.0)
            }
        }

        impl<T> $name<T> {
            /// Construct a vector by setting all lanes to the given value.
            #[inline]
            pub fn splat(value: *$mut T) -> Self {
                Self(<$underlying>::splat(value as isize), PhantomData)
            }
            call_counting_args! { $lanes => define_pointer_vector => new $underlying | *$mut T | }
        }

        // array references
        impl<T> AsRef<[*$mut T; $lanes]> for $name<T> {
            #[inline]
            fn as_ref(&self) -> &[*$mut T; $lanes] {
                unsafe { &*(self as *const _ as *const _) }
            }
        }

        impl<T> AsMut<[*$mut T; $lanes]> for $name<T> {
            #[inline]
            fn as_mut(&mut self) -> &mut [*$mut T; $lanes] {
                unsafe { &mut *(self as *mut _ as *mut _) }
            }
        }

        // slice references
        impl<T> AsRef<[*$mut T]> for $name<T> {
            #[inline]
            fn as_ref(&self) -> &[*$mut T] {
                AsRef::<[*$mut T; $lanes]>::as_ref(self)
            }
        }

        impl<T> AsMut<[*$mut T]> for $name<T> {
            #[inline]
            fn as_mut(&mut self) -> &mut [*$mut T] {
                AsMut::<[*$mut T; $lanes]>::as_mut(self)
            }
        }

        // splat
        impl<T> From<*$mut T> for $name<T> {
            #[inline]
            fn from(value: *$mut T) -> Self {
                Self::splat(value)
            }
        }
    };
    { new $underlying:ty | $type:ty | $($var:ident)* } => {
        /// Construct a vector by setting each lane to the given values.
        #[allow(clippy::too_many_arguments)]
        #[inline]
        pub fn new($($var: $type),*) -> Self {
            Self(<$underlying>::new($($var as isize),*), PhantomData)
        }
    };
}

define_pointer_vector! { #[doc = "Vector of two mutable pointers"] mptrx2 => isizex2 => 2, mut }
define_pointer_vector! { #[doc = "Vector of four mutable pointers"] mptrx4 => isizex4 => 4, mut }
define_pointer_vector! { #[doc = "Vector of eight mutable pointers"] mptrx8 => isizex8 => 8, mut }
define_pointer_vector! { #[doc = "Vector of two const pointers"] cptrx2 => isizex2 => 2, const }
define_pointer_vector! { #[doc = "Vector of four const pointers"] cptrx4 => isizex4 => 4, const }
define_pointer_vector! { #[doc = "Vector of eight const pointers"] cptrx8 => isizex8 => 8, const }
