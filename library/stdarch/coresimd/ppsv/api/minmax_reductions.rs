//! Implements portable arithmetic vector reductions.
#![allow(unused)]

macro_rules! impl_minmax_reductions {
    ($id:ident, $elem_ty:ident) => {
        impl $id {
            /// Largest vector value.
            ///
            /// FIXME: document behavior for float vectors with NaNs.
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn max(self) -> $elem_ty {
                use ::coresimd::simd_llvm::simd_reduce_max;
                unsafe {
                    simd_reduce_max(self)
                }
            }
            /// Largest vector value.
            ///
            /// FIXME: document behavior for float vectors with NaNs.
            #[cfg(target_arch = "aarch64")]
            #[allow(unused_imports)]
            #[inline]
            pub fn max(self) -> $elem_ty {
                // FIXME: broken on AArch64
                use ::num::Float;
                use ::cmp::Ord;
                let mut x = self.extract(0);
                for i in 1..$id::lanes() {
                    x = x.max(self.extract(i));
                }
                x
            }

            /// Smallest vector value.
            ///
            /// FIXME: document behavior for float vectors with NaNs.
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn min(self) -> $elem_ty {
                use ::coresimd::simd_llvm::simd_reduce_min;
                unsafe {
                    simd_reduce_min(self)
                }
            }
            /// Smallest vector value.
            ///
            /// FIXME: document behavior for float vectors with NaNs.
            #[cfg(target_arch = "aarch64")]
            #[allow(unused_imports)]
            #[inline]
            pub fn min(self) -> $elem_ty {
                // FIXME: broken on AArch64
                use ::num::Float;
                use ::cmp::Ord;
                let mut x = self.extract(0);
                for i in 1..$id::lanes() {
                    x = x.min(self.extract(i));
                }
                x
            }
        }
    }
}

#[cfg(test)]
macro_rules! test_minmax_reductions {
    ($id:ident, $elem_ty:ident) => {
        #[test]
        fn max() {
            use ::coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.max(), 0 as $elem_ty);
            let v = v.replace(1, 1 as $elem_ty);
            assert_eq!(v.max(), 1 as $elem_ty);
            let v = v.replace(0, 2 as $elem_ty);
            assert_eq!(v.max(), 2 as $elem_ty);
        }

        #[test]
        fn min() {
            use ::coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.min(), 0 as $elem_ty);
            let v = v.replace(1, 1 as $elem_ty);
            assert_eq!(v.min(), 0 as $elem_ty);
            let v = $id::splat(1 as $elem_ty);
            let v = v.replace(0, 2 as $elem_ty);
            assert_eq!(v.min(), 1 as $elem_ty);
            let v = $id::splat(2 as $elem_ty);
            let v = v.replace(1, 1 as $elem_ty);
            assert_eq!(v.min(), 1 as $elem_ty);
        }
    }
}
