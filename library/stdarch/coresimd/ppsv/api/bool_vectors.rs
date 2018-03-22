//! Minimal boolean vector implementation
#![allow(unused)]

/// Minimal interface: all packed SIMD boolean vector types implement this.
macro_rules! impl_bool_minimal {
    ($id:ident, $elem_ty:ident, $elem_count:expr, $($elem_name:ident),+) => {
        impl $id {
            /// Creates a new instance with each vector elements initialized
            /// with the provided values.
            #[inline]
            pub const fn new($($elem_name: bool),*) -> Self {
                $id($(Self::bool_to_internal($elem_name)),*)
            }

            /// Converts a boolean type into the type of the vector lanes.
            #[inline]
            const fn bool_to_internal(x: bool) -> $elem_ty {
                [0 as $elem_ty, !(0 as $elem_ty)][x as usize]
            }

            /// Returns the number of vector lanes.
            #[inline]
            pub const fn lanes() -> usize {
                $elem_count
            }

            /// Constructs a new instance with each element initialized to
            /// `value`.
            #[inline]
            pub const fn splat(value: bool) -> Self {
                $id($({
                    #[allow(non_camel_case_types, dead_code)]
                    struct $elem_name;
                    Self::bool_to_internal(value)
                }),*)
            }

            /// Extracts the value at `index`.
            ///
            /// # Panics
            ///
            /// If `index >= Self::lanes()`.
            #[inline]
            pub fn extract(self, index: usize) -> bool {
                assert!(index < $elem_count);
                unsafe { self.extract_unchecked(index) }
            }

            /// Extracts the value at `index`.
            ///
            /// If `index >= Self::lanes()` the behavior is undefined.
            #[inline]
            pub unsafe fn extract_unchecked(self, index: usize) -> bool {
                use coresimd::simd_llvm::simd_extract;
                let x: $elem_ty = simd_extract(self, index as u32);
                x != 0
            }

            /// Returns a new vector where the value at `index` is replaced by `new_value`.
            ///
            /// # Panics
            ///
            /// If `index >= Self::lanes()`.
            #[inline]
            #[must_use = "replace does not modify the original value - it returns a new vector with the value at `index` replaced by `new_value`d"]
            pub fn replace(self, index: usize, new_value: bool) -> Self {
                assert!(index < $elem_count);
                unsafe { self.replace_unchecked(index, new_value) }
            }

            /// Returns a new vector where the value at `index` is replaced by `new_value`.
            ///
            /// # Panics
            ///
            /// If `index >= Self::lanes()`.
            #[inline]
            #[must_use = "replace_unchecked does not modify the original value - it returns a new vector with the value at `index` replaced by `new_value`d"]
            pub unsafe fn replace_unchecked(
                self,
                index: usize,
                new_value: bool,
            ) -> Self {
                use coresimd::simd_llvm::simd_insert;
                simd_insert(self, index as u32, Self::bool_to_internal(new_value))
            }
        }
    }
}

#[cfg(test)]
macro_rules! test_bool_minimal {
    ($id: ident, $elem_count: expr) => {
        #[test]
        fn minimal() {
            use coresimd::simd::$id;
            // TODO: test new

            // lanes:
            assert_eq!($elem_count, $id::lanes());

            // splat and extract / extract_unchecked:
            let vec = $id::splat(true);
            for i in 0..$id::lanes() {
                assert_eq!(true, vec.extract(i));
                assert_eq!(true, unsafe { vec.extract_unchecked(i) });
            }

            // replace / replace_unchecked
            let new_vec = vec.replace(1, false);
            for i in 0..$id::lanes() {
                if i == 1 {
                    assert_eq!(false, new_vec.extract(i));
                } else {
                    assert_eq!(true, new_vec.extract(i));
                }
            }
            let new_vec = unsafe { vec.replace_unchecked(1, false) };
            for i in 0..$id::lanes() {
                if i == 1 {
                    assert_eq!(false, new_vec.extract(i));
                } else {
                    assert_eq!(true, new_vec.extract(i));
                }
            }
        }
        #[test]
        #[should_panic]
        fn minimal_extract_panic_on_out_of_bounds() {
            use coresimd::simd::$id;
            let vec = $id::splat(false);
            let _ = vec.extract($id::lanes());
        }
        #[test]
        #[should_panic]
        fn minimal_replace_panic_on_out_of_bounds() {
            use coresimd::simd::$id;
            let vec = $id::splat(false);
            let _ = vec.replace($id::lanes(), true);
        }
    };
}
