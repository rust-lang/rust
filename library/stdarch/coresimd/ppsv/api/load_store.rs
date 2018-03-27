//! Implements the load/store API.
#![allow(unused)]

macro_rules! impl_load_store {
    ($id: ident, $elem_ty: ident, $elem_count: expr) => {
        impl $id {
            /// Writes the values of the vector to the `slice`.
            ///
            /// # Panics
            ///
            /// If `slice.len() < Self::lanes()` or `&slice[0]` is not
            /// aligned to an `align_of::<Self>()` boundary.
            #[inline]
            pub fn store_aligned(self, slice: &mut [$elem_ty]) {
                use slice::SliceExt;
                unsafe {
                    assert!(slice.len() >= $elem_count);
                    let target_ptr =
                        slice.get_unchecked_mut(0) as *mut $elem_ty;
                    assert!(
                        target_ptr.align_offset(::mem::align_of::<Self>())
                            == 0
                    );
                    self.store_aligned_unchecked(slice);
                }
            }

            /// Writes the values of the vector to the `slice`.
            ///
            /// # Panics
            ///
            /// If `slice.len() < Self::lanes()`.
            #[inline]
            pub fn store_unaligned(self, slice: &mut [$elem_ty]) {
                use slice::SliceExt;
                unsafe {
                    assert!(slice.len() >= $elem_count);
                    self.store_unaligned_unchecked(slice);
                }
            }

            /// Writes the values of the vector to the `slice`.
            ///
            /// # Precondition
            ///
            /// If `slice.len() < Self::lanes()` or `&slice[0]` is not
            /// aligned to an `align_of::<Self>()` boundary, the behavior is
            /// undefined.
            #[inline]
            pub unsafe fn store_aligned_unchecked(
                self, slice: &mut [$elem_ty]
            ) {
                use slice::SliceExt;
                *(slice.get_unchecked_mut(0) as *mut $elem_ty as *mut Self) =
                    self;
            }

            /// Writes the values of the vector to the `slice`.
            ///
            /// # Precondition
            ///
            /// If `slice.len() < Self::lanes()` the behavior is undefined.
            #[inline]
            pub unsafe fn store_unaligned_unchecked(
                self, slice: &mut [$elem_ty]
            ) {
                use slice::SliceExt;
                let target_ptr =
                    slice.get_unchecked_mut(0) as *mut $elem_ty as *mut u8;
                let self_ptr = &self as *const Self as *const u8;
                ::ptr::copy_nonoverlapping(
                    self_ptr,
                    target_ptr,
                    ::mem::size_of::<Self>(),
                );
            }

            /// Instantiates a new vector with the values of the `slice`.
            ///
            /// # Panics
            ///
            /// If `slice.len() < Self::lanes()` or `&slice[0]` is not aligned
            /// to an `align_of::<Self>()` boundary.
            #[inline]
            pub fn load_aligned(slice: &[$elem_ty]) -> Self {
                unsafe {
                    use slice::SliceExt;
                    assert!(slice.len() >= $elem_count);
                    let target_ptr = slice.get_unchecked(0) as *const $elem_ty;
                    assert!(
                        target_ptr.align_offset(::mem::align_of::<Self>())
                            == 0
                    );
                    Self::load_aligned_unchecked(slice)
                }
            }

            /// Instantiates a new vector with the values of the `slice`.
            ///
            /// # Panics
            ///
            /// If `slice.len() < Self::lanes()`.
            #[inline]
            pub fn load_unaligned(slice: &[$elem_ty]) -> Self {
                use slice::SliceExt;
                unsafe {
                    assert!(slice.len() >= $elem_count);
                    Self::load_unaligned_unchecked(slice)
                }
            }

            /// Instantiates a new vector with the values of the `slice`.
            ///
            /// # Precondition
            ///
            /// If `slice.len() < Self::lanes()` or `&slice[0]` is not aligned
            /// to an `align_of::<Self>()` boundary, the behavior is undefined.
            #[inline]
            pub unsafe fn load_aligned_unchecked(slice: &[$elem_ty]) -> Self {
                use slice::SliceExt;
                *(slice.get_unchecked(0) as *const $elem_ty as *const Self)
            }

            /// Instantiates a new vector with the values of the `slice`.
            ///
            /// # Precondition
            ///
            /// If `slice.len() < Self::lanes()` the behavior is undefined.
            #[inline]
            pub unsafe fn load_unaligned_unchecked(
                slice: &[$elem_ty]
            ) -> Self {
                use mem::size_of;
                use slice::SliceExt;
                let target_ptr =
                    slice.get_unchecked(0) as *const $elem_ty as *const u8;
                let mut x = Self::splat(0 as $elem_ty);
                let self_ptr = &mut x as *mut Self as *mut u8;
                ::ptr::copy_nonoverlapping(
                    target_ptr,
                    self_ptr,
                    size_of::<Self>(),
                );
                x
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_load_store {
    ($id: ident, $elem_ty: ident) => {
        #[test]
        fn store_unaligned() {
            use coresimd::simd::$id;
            use std::iter::Iterator;
            let mut unaligned = [0 as $elem_ty; $id::lanes() + 1];
            let vec = $id::splat(42 as $elem_ty);
            vec.store_unaligned(&mut unaligned[1..]);
            for (index, &b) in unaligned.iter().enumerate() {
                if index == 0 {
                    assert_eq!(b, 0 as $elem_ty);
                } else {
                    assert_eq!(b, vec.extract(index - 1));
                }
            }
        }

        #[test]
        #[should_panic]
        fn store_unaligned_fail() {
            use coresimd::simd::$id;
            let mut unaligned = [0 as $elem_ty; $id::lanes() + 1];
            let vec = $id::splat(42 as $elem_ty);
            vec.store_unaligned(&mut unaligned[2..]);
        }

        #[test]
        fn load_unaligned() {
            use coresimd::simd::$id;
            use std::iter::Iterator;
            let mut unaligned = [42 as $elem_ty; $id::lanes() + 1];
            unaligned[0] = 0 as $elem_ty;
            let vec = $id::load_unaligned(&unaligned[1..]);
            for (index, &b) in unaligned.iter().enumerate() {
                if index == 0 {
                    assert_eq!(b, 0 as $elem_ty);
                } else {
                    assert_eq!(b, vec.extract(index - 1));
                }
            }
        }

        #[test]
        #[should_panic]
        fn load_unaligned_fail() {
            use coresimd::simd::$id;
            let mut unaligned = [42 as $elem_ty; $id::lanes() + 1];
            unaligned[0] = 0 as $elem_ty;
            let _vec = $id::load_unaligned(&unaligned[2..]);
        }

        union A {
            data: [$elem_ty; 2 * ::coresimd::simd::$id::lanes()],
            _vec: ::coresimd::simd::$id,
        }

        #[test]
        fn store_aligned() {
            use coresimd::simd::$id;
            use std::iter::Iterator;
            let mut aligned = A {
                data: [0 as $elem_ty; 2 * $id::lanes()],
            };
            let vec = $id::splat(42 as $elem_ty);
            unsafe { vec.store_aligned(&mut aligned.data[$id::lanes()..]) };
            for (index, &b) in unsafe { aligned.data.iter().enumerate() } {
                if index < $id::lanes() {
                    assert_eq!(b, 0 as $elem_ty);
                } else {
                    assert_eq!(b, vec.extract(index - $id::lanes()));
                }
            }
        }

        #[test]
        #[should_panic]
        fn store_aligned_fail_lanes() {
            use coresimd::simd::$id;
            let mut aligned = A {
                data: [0 as $elem_ty; 2 * $id::lanes()],
            };
            let vec = $id::splat(42 as $elem_ty);
            unsafe {
                vec.store_aligned(&mut aligned.data[2 * $id::lanes()..])
            };
        }

        #[test]
        #[should_panic]
        fn store_aligned_fail_align() {
            unsafe {
                use coresimd::simd::$id;
                use std::{mem, slice};
                let mut aligned = A {
                    data: [0 as $elem_ty; 2 * $id::lanes()],
                };
                // offset the aligned data by one byte:
                let s: &mut [u8; 2 * $id::lanes()
                                * mem::size_of::<$elem_ty>()] =
                    mem::transmute(&mut aligned.data);
                let s: &mut [$elem_ty] = slice::from_raw_parts_mut(
                    s.get_unchecked_mut(1) as *mut u8 as *mut $elem_ty,
                    $id::lanes(),
                );
                let vec = $id::splat(42 as $elem_ty);
                vec.store_aligned(s);
            }
        }

        #[test]
        fn load_aligned() {
            use coresimd::simd::$id;
            use std::iter::Iterator;
            let mut aligned = A {
                data: [0 as $elem_ty; 2 * $id::lanes()],
            };
            for i in $id::lanes()..(2 * $id::lanes()) {
                unsafe {
                    aligned.data[i] = 42 as $elem_ty;
                }
            }

            let vec =
                unsafe { $id::load_aligned(&aligned.data[$id::lanes()..]) };
            for (index, &b) in unsafe { aligned.data.iter().enumerate() } {
                if index < $id::lanes() {
                    assert_eq!(b, 0 as $elem_ty);
                } else {
                    assert_eq!(b, vec.extract(index - $id::lanes()));
                }
            }
        }

        #[test]
        #[should_panic]
        fn load_aligned_fail_lanes() {
            use coresimd::simd::$id;
            let aligned = A {
                data: [0 as $elem_ty; 2 * $id::lanes()],
            };
            let _vec = unsafe {
                $id::load_aligned(&aligned.data[2 * $id::lanes()..])
            };
        }

        #[test]
        #[should_panic]
        fn load_aligned_fail_align() {
            unsafe {
                use coresimd::simd::$id;
                use std::{mem, slice};
                let aligned = A {
                    data: [0 as $elem_ty; 2 * $id::lanes()],
                };
                // offset the aligned data by one byte:
                let s: &[u8; 2 * $id::lanes()
                            * mem::size_of::<$elem_ty>()] =
                    mem::transmute(&aligned.data);
                let s: &[$elem_ty] = slice::from_raw_parts(
                    s.get_unchecked(1) as *const u8 as *const $elem_ty,
                    $id::lanes(),
                );
                let _vec = $id::load_aligned(s);
            }
        }
    };
}
