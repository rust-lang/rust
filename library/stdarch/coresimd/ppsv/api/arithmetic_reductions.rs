//! Implements portable arithmetic vector reductions.
#![allow(unused)]

macro_rules! impl_arithmetic_reductions {
    ($id:ident, $elem_ty:ident) => {
        impl $id {
            /// Horizontal sum of the vector elements.
            ///
            /// The intrinsic performs a tree-reduction of the vector elements.
            /// That is, for an 8 element vector:
            ///
            /// > ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7))
            ///
            /// # Integer vectors
            ///
            /// If an operation overflows it returns the mathematical result
            /// modulo `2^n` where `n` is the number of times it overflows.
            ///
            /// # Floating-point vectors
            ///
            /// If one of the vector element is `NaN` the reduction returns
            /// `NaN`.
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn wrapping_sum(self) -> $elem_ty {
                use coresimd::simd_llvm::simd_reduce_add_ordered;
                unsafe { simd_reduce_add_ordered(self, 0 as $elem_ty) }
            }
            /// Horizontal sum of the vector elements.
            ///
            /// The intrinsic performs a tree-reduction of the vector elements.
            /// That is, for an 8 element vector:
            ///
            /// > ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7))
            ///
            /// # Integer vectors
            ///
            /// If an operation overflows it returns the mathematical result
            /// modulo `2^n` where `n` is the number of times it overflows.
            ///
            /// # Floating-point vectors
            ///
            /// If one of the vector element is `NaN` the reduction returns
            /// `NaN`.
            #[cfg(target_arch = "aarch64")]
            #[inline]
            pub fn wrapping_sum(self) -> $elem_ty {
                // FIXME: broken on AArch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                use super::codegen::wrapping::Wrapping;
                let mut x = self.extract(0) as $elem_ty;
                for i in 1..$id::lanes() {
                    x = Wrapping::add(x, self.extract(i) as $elem_ty);
                }
                x
            }

            /// Horizontal product of the vector elements.
            ///
            /// The intrinsic performs a tree-reduction of the vector elements.
            /// That is, for an 8 element vector:
            ///
            /// > ((x0 * x1) * (x2 * x3)) * ((x4 * x5) * (x6 * x7))
            ///
            /// # Integer vectors
            ///
            /// If an operation overflows it returns the mathematical result
            /// modulo `2^n` where `n` is the number of times it overflows.
            ///
            /// # Floating-point vectors
            ///
            /// If one of the vector element is `NaN` the reduction returns
            /// `NaN`.
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn wrapping_product(self) -> $elem_ty {
                use coresimd::simd_llvm::simd_reduce_mul_ordered;
                unsafe { simd_reduce_mul_ordered(self, 1 as $elem_ty) }
            }
            /// Horizontal product of the vector elements.
            ///
            /// The intrinsic performs a tree-reduction of the vector elements.
            /// That is, for an 8 element vector:
            ///
            /// > ((x0 * x1) * (x2 * x3)) * ((x4 * x5) * (x6 * x7))
            ///
            /// # Integer vectors
            ///
            /// If an operation overflows it returns the mathematical result
            /// modulo `2^n` where `n` is the number of times it overflows.
            ///
            /// # Floating-point vectors
            ///
            /// If one of the vector element is `NaN` the reduction returns
            /// `NaN`.
            #[cfg(target_arch = "aarch64")]
            #[inline]
            pub fn wrapping_product(self) -> $elem_ty {
                // FIXME: broken on AArch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                use super::codegen::wrapping::Wrapping;
                let mut x = self.extract(0) as $elem_ty;
                for i in 1..$id::lanes() {
                    x = Wrapping::mul(x, self.extract(i) as $elem_ty);
                }
                x
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_arithmetic_reductions {
    ($id:ident, $elem_ty:ident) => {
        fn alternating(x: usize) -> ::coresimd::simd::$id {
            use coresimd::simd::$id;
            let mut v = $id::splat(1 as $elem_ty);
            for i in 0..$id::lanes() {
                if i % x == 0 {
                    v = v.replace(i, 2 as $elem_ty);
                }
            }
            v
        }

        #[test]
        fn wrapping_sum() {
            use coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.wrapping_sum(), 0 as $elem_ty);
            let v = $id::splat(1 as $elem_ty);
            assert_eq!(v.wrapping_sum(), $id::lanes() as $elem_ty);
            let v = alternating(2);
            assert_eq!(
                v.wrapping_sum(),
                ($id::lanes() / 2 + $id::lanes()) as $elem_ty
            );
        }
        #[test]
        fn wrapping_product() {
            use coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.wrapping_product(), 0 as $elem_ty);
            let v = $id::splat(1 as $elem_ty);
            assert_eq!(v.wrapping_product(), 1 as $elem_ty);
            let f = match $id::lanes() {
                64 => 16,
                32 => 8,
                16 => 4,
                _ => 2,
            };
            let v = alternating(f);
            assert_eq!(
                v.wrapping_product(),
                (2_usize.pow(($id::lanes() / f) as u32) as $elem_ty)
            );
        }
    };
}
