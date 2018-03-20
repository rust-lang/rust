//! Minimal portable vector types API.
#![allow(unused)]

/// Minimal interface: all packed SIMD vector types implement this.
macro_rules! impl_minimal {
    ($id:ident, $elem_ty:ident, $elem_count:expr, $($elem_name:ident),+) => {
        impl $id {
            /// Creates a new instance with each vector elements initialized
            /// with the provided values.
            #[inline]
            pub const fn new($($elem_name: $elem_ty),*) -> Self {
                $id($($elem_name),*)
            }

            /// Returns the number of vector lanes.
            #[inline]
            pub const fn lanes() -> usize {
                $elem_count
            }

            /// Constructs a new instance with each element initialized to
            /// `value`.
            #[inline]
            pub const fn splat(value: $elem_ty) -> Self {
                $id($({
                    #[allow(non_camel_case_types, dead_code)]
                    struct $elem_name;
                    value
                }),*)
            }

            /// Extracts the value at `index`.
            ///
            /// # Panics
            ///
            /// If `index >= Self::lanes()`.
            #[inline]
            pub fn extract(self, index: usize) -> $elem_ty {
                assert!(index < $elem_count);
                unsafe { self.extract_unchecked(index) }
            }

            /// Extracts the value at `index`.
            ///
            /// # Precondition
            ///
            /// If `index >= Self::lanes()` the behavior is undefined.
            #[inline]
            pub unsafe fn extract_unchecked(self, index: usize) -> $elem_ty {
                use coresimd::simd_llvm::simd_extract;
                simd_extract(self, index as u32)
            }

            /// Returns a new vector where the value at `index` is replaced by `new_value`.
            ///
            /// # Panics
            ///
            /// If `index >= Self::lanes()`.
            #[inline]
            #[must_use = "replace does not modify the original value - it returns a new vector with the value at `index` replaced by `new_value`d"]
            pub fn replace(self, index: usize, new_value: $elem_ty) -> Self {
                assert!(index < $elem_count);
                unsafe { self.replace_unchecked(index, new_value) }
            }

            /// Returns a new vector where the value at `index` is replaced by `new_value`.
            ///
            /// # Precondition
            ///
            /// If `index >= Self::lanes()` the behavior is undefined.
            #[inline]
            #[must_use = "replace_unchecked does not modify the original value - it returns a new vector with the value at `index` replaced by `new_value`d"]
            pub unsafe fn replace_unchecked(
                self,
                index: usize,
                new_value: $elem_ty,
            ) -> Self {
                use coresimd::simd_llvm::simd_insert;
                simd_insert(self, index as u32, new_value)
            }
        }
    }
}

#[cfg(test)]
macro_rules! test_minimal {
    ($id:ident, $elem_ty:ident, $elem_count:expr) => {
        #[test]
        fn minimal() {
            use ::coresimd::simd::$id;
            // TODO: test new

            // lanes:
            assert_eq!($elem_count, $id::lanes());

            // splat and extract / extract_unchecked:
            const VAL: $elem_ty = 7 as $elem_ty;
            const VEC: $id = $id::splat(VAL);
            for i in 0..$id::lanes() {
                assert_eq!(VAL, VEC.extract(i));
                assert_eq!(VAL, unsafe { VEC.extract_unchecked(i) });
            }

            // replace / replace_unchecked
            let new_vec = VEC.replace(1, 42 as $elem_ty);
            for i in 0..$id::lanes() {
                if i == 1 {
                    assert_eq!(42 as $elem_ty, new_vec.extract(i));
                } else {
                    assert_eq!(VAL, new_vec.extract(i));
                }
            }
            let new_vec = unsafe { VEC.replace_unchecked(1, 42 as $elem_ty) };
            for i in 0..$id::lanes() {
                if i == 1 {
                    assert_eq!(42 as $elem_ty, new_vec.extract(i));
                } else {
                    assert_eq!(VAL, new_vec.extract(i));
                }
            }
        }
        #[test]
        #[should_panic]
        fn minimal_extract_panic_on_out_of_bounds() {
            use ::coresimd::simd::$id;
            const VAL: $elem_ty = 7 as $elem_ty;
            const VEC: $id = $id::splat(VAL);
            let _ = VEC.extract($id::lanes());
        }
        #[test]
        #[should_panic]
        fn minimal_replace_panic_on_out_of_bounds() {
            use ::coresimd::simd::$id;
            const VAL: $elem_ty = 7 as $elem_ty;
            const VEC: $id = $id::splat(VAL);
            let _ = VEC.replace($id::lanes(), 42 as $elem_ty);
        }
    }
}
