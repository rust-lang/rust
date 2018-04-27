//! Implements portable horizontal arithmetic reductions.
#![allow(unused)]

macro_rules! impl_minmax_reductions {
    ($id:ident, $elem_ty:ident) => {
        impl $id {
            /// Largest vector element value.
            #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
            #[inline]
            pub fn max_element(self) -> $elem_ty {
                use coresimd::simd_llvm::simd_reduce_max;
                unsafe { simd_reduce_max(self) }
            }

            /// Largest vector element value.
            #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
            #[allow(unused_imports)]
            #[inline]
            pub fn max_element(self) -> $elem_ty {
                // FIXME: broken on AArch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                use cmp::Ord;
                let mut x = self.extract(0);
                for i in 1..$id::lanes() {
                    x = x.max(self.extract(i));
                }
                x
            }

            /// Smallest vector element value.
            #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
            #[inline]
            pub fn min_element(self) -> $elem_ty {
                use coresimd::simd_llvm::simd_reduce_min;
                unsafe { simd_reduce_min(self) }
            }

            /// Smallest vector element value.
            #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
            #[allow(unused_imports)]
            #[inline]
            pub fn min_element(self) -> $elem_ty {
                // FIXME: broken on AArch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                use cmp::Ord;
                let mut x = self.extract(0);
                for i in 1..$id::lanes() {
                    x = x.min(self.extract(i));
                }
                x
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_minmax_reductions {
    ($id:ident, $elem_ty:ident) => {
        #[test]
        fn max_element() {
            use coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.max_element(), 0 as $elem_ty);
            let v = v.replace(1, 1 as $elem_ty);
            assert_eq!(v.max_element(), 1 as $elem_ty);
            let v = v.replace(0, 2 as $elem_ty);
            assert_eq!(v.max_element(), 2 as $elem_ty);
        }

        #[test]
        fn min_element() {
            use coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.min_element(), 0 as $elem_ty);
            let v = v.replace(1, 1 as $elem_ty);
            assert_eq!(v.min_element(), 0 as $elem_ty);
            let v = $id::splat(1 as $elem_ty);
            let v = v.replace(0, 2 as $elem_ty);
            assert_eq!(v.min_element(), 1 as $elem_ty);
            let v = $id::splat(2 as $elem_ty);
            let v = v.replace(1, 1 as $elem_ty);
            assert_eq!(v.min_element(), 1 as $elem_ty);
        }
    };
}
