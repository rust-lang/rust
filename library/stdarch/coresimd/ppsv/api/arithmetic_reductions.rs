//! Implements portable arithmetic vector reductions.
#![allow(unused)]

macro_rules! impl_arithmetic_reductions {
    ($id:ident, $elem_ty:ident) => {
        impl $id {
            /// Lane-wise addition of the vector elements.
            ///
            /// FIXME: document guarantees with respect to:
            ///    * integers: overflow behavior
            ///    * floats: order and NaNs
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn sum(self) -> $elem_ty {
                use ::coresimd::simd_llvm::simd_reduce_add_ordered;
                unsafe {
                    simd_reduce_add_ordered(self, 0 as $elem_ty)
                }
            }
            /// Lane-wise addition of the vector elements.
            ///
            /// FIXME: document guarantees with respect to:
            ///    * integers: overflow behavior
            ///    * floats: order and NaNs
            #[cfg(target_arch = "aarch64")]
            #[inline]
            pub fn sum(self) -> $elem_ty {
                // FIXME: broken on AArch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                let mut x = self.extract(0) as $elem_ty;
                for i in 1..$id::lanes() {
                    x += self.extract(i) as $elem_ty;
                }
                x
            }

            /// Lane-wise multiplication of the vector elements.
            ///
            /// FIXME: document guarantees with respect to:
            ///    * integers: overflow behavior
            ///    * floats: order and NaNs
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn product(self) -> $elem_ty {
                use ::coresimd::simd_llvm::simd_reduce_mul_ordered;
                unsafe {
                    simd_reduce_mul_ordered(self, 1 as $elem_ty)
                }
            }
            /// Lane-wise multiplication of the vector elements.
            ///
            /// FIXME: document guarantees with respect to:
            ///    * integers: overflow behavior
            ///    * floats: order and NaNs
            #[cfg(target_arch = "aarch64")]
            #[inline]
            pub fn product(self) -> $elem_ty {
                // FIXME: broken on AArch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                let mut x = self.extract(0) as $elem_ty;
                for i in 1..$id::lanes() {
                    x *= self.extract(i) as $elem_ty;
                }
                x
            }
        }
    }
}

#[cfg(test)]
macro_rules! test_arithmetic_reductions {
    ($id:ident, $elem_ty:ident) => {

        fn alternating(x: usize) -> ::coresimd::simd::$id {
            use ::coresimd::simd::$id;
            let mut v = $id::splat(1 as $elem_ty);
            for i in 0..$id::lanes() {
                if i % x == 0 {
                    v = v.replace(i, 2 as $elem_ty);
                }
            }
            v
        }

        #[test]
        fn sum() {
            use ::coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.sum(), 0 as $elem_ty);
            let v = $id::splat(1 as $elem_ty);
            assert_eq!(v.sum(), $id::lanes() as $elem_ty);
            let v = alternating(2);
            assert_eq!(v.sum(), ($id::lanes() / 2 + $id::lanes()) as $elem_ty);
        }
        #[test]
        fn product() {
            use ::coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.product(), 0 as $elem_ty);
            let v = $id::splat(1 as $elem_ty);
            assert_eq!(v.product(), 1 as $elem_ty);
            let f = match $id::lanes() {
                64 => 16,
                32 => 8,
                16 => 4,
                _ => 2,
            };
            let v = alternating(f);
            assert_eq!(v.product(), (2_usize.pow(($id::lanes() / f) as u32) as $elem_ty));
        }
    }
}
