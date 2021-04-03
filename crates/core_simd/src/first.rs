/// Implements common traits on the specified vector `$name`, holding multiple `$lanes` of `$type`.
macro_rules! impl_vector {
    { $name:ident, $type:ty } => {
        impl<const LANES: usize> $name<LANES> where Self: crate::LanesAtMost32 {
            /// Construct a SIMD vector by setting all lanes to the given value.
            pub const fn splat(value: $type) -> Self {
                Self([value; LANES])
            }

            /// Returns a slice containing the entire SIMD vector.
            pub const fn as_slice(&self) -> &[$type] {
                &self.0
            }

            /// Returns a mutable slice containing the entire SIMD vector.
            pub fn as_mut_slice(&mut self) -> &mut [$type] {
                &mut self.0
            }

            /// Converts an array to a SIMD vector.
            pub const fn from_array(array: [$type; LANES]) -> Self {
                Self(array)
            }

            /// Converts a SIMD vector to an array.
            pub const fn to_array(self) -> [$type; LANES] {
                // workaround for rust-lang/rust#80108
                // TODO fix this
                #[cfg(target_arch = "wasm32")]
                {
                    let mut arr = [self.0[0]; LANES];
                    let mut i = 0;
                    while i < LANES {
                        arr[i] = self.0[i];
                        i += 1;
                    }
                    arr
                }

                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.0
                }
            }
        }

        impl<const LANES: usize> Copy for $name<LANES> where Self: crate::LanesAtMost32 {}

        impl<const LANES: usize> Clone for $name<LANES> where Self: crate::LanesAtMost32 {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<const LANES: usize> Default for $name<LANES> where Self: crate::LanesAtMost32 {
            #[inline]
            fn default() -> Self {
                Self::splat(<$type>::default())
            }
        }

        impl<const LANES: usize> PartialEq for $name<LANES> where Self: crate::LanesAtMost32 {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                // TODO use SIMD equality
                self.to_array() == other.to_array()
            }
        }

        impl<const LANES: usize> PartialOrd for $name<LANES> where Self: crate::LanesAtMost32 {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                // TODO use SIMD equalitya
                self.to_array().partial_cmp(other.as_ref())
            }
        }

        // array references
        impl<const LANES: usize> AsRef<[$type; LANES]> for $name<LANES> where Self: crate::LanesAtMost32 {
            #[inline]
            fn as_ref(&self) -> &[$type; LANES] {
                &self.0
            }
        }

        impl<const LANES: usize> AsMut<[$type; LANES]> for $name<LANES> where Self: crate::LanesAtMost32 {
            #[inline]
            fn as_mut(&mut self) -> &mut [$type; LANES] {
                &mut self.0
            }
        }

        // slice references
        impl<const LANES: usize> AsRef<[$type]> for $name<LANES> where Self: crate::LanesAtMost32 {
            #[inline]
            fn as_ref(&self) -> &[$type] {
                &self.0
            }
        }

        impl<const LANES: usize> AsMut<[$type]> for $name<LANES> where Self: crate::LanesAtMost32 {
            #[inline]
            fn as_mut(&mut self) -> &mut [$type] {
                &mut self.0
            }
        }

        // vector/array conversion
        impl<const LANES: usize> From<[$type; LANES]> for $name<LANES> where Self: crate::LanesAtMost32 {
            fn from(array: [$type; LANES]) -> Self {
                Self(array)
            }
        }

        impl <const LANES: usize> From<$name<LANES>> for [$type; LANES] where $name<LANES>: crate::LanesAtMost32 {
            fn from(vector: $name<LANES>) -> Self {
                vector.to_array()
            }
        }

        impl_shuffle_2pow_lanes!{ $name }
    }
}
