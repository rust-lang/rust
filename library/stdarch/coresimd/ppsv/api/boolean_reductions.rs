//! Lane-wise boolean vector reductions.
#![allow(unused)]

macro_rules! impl_bool_reductions {
    ($id: ident) => {
        impl $id {
            /// Are `all` vector lanes `true`?
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn all(self) -> bool {
                use coresimd::simd_llvm::simd_reduce_all;
                unsafe { simd_reduce_all(self) }
            }
            /// Are `all` vector lanes `true`?
            #[cfg(target_arch = "aarch64")]
            #[inline]
            pub fn all(self) -> bool {
                // FIXME: Broken on AArch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                self.and()
            }

            /// Is `any` vector lanes `true`?
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn any(self) -> bool {
                use coresimd::simd_llvm::simd_reduce_any;
                unsafe { simd_reduce_any(self) }
            }
            /// Is `any` vector lanes `true`?
            #[cfg(target_arch = "aarch64")]
            #[inline]
            pub fn any(self) -> bool {
                // FIXME: Broken on AArch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                self.or()
            }

            /// Are `all` vector lanes `false`?
            #[inline]
            pub fn none(self) -> bool {
                !self.any()
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_bool_reductions {
    ($id: ident) => {
        #[test]
        fn all() {
            use coresimd::simd::$id;

            let a = $id::splat(true);
            assert!(a.all());
            let a = $id::splat(false);
            assert!(!a.all());

            for i in 0..$id::lanes() {
                let mut a = $id::splat(true);
                a = a.replace(i, false);
                assert!(!a.all());
                let mut a = $id::splat(false);
                a = a.replace(i, true);
                assert!(!a.all());
            }
        }
        #[test]
        fn any() {
            use coresimd::simd::$id;

            let a = $id::splat(true);
            assert!(a.any());
            let a = $id::splat(false);
            assert!(!a.any());

            for i in 0..$id::lanes() {
                let mut a = $id::splat(true);
                a = a.replace(i, false);
                assert!(a.any());
                let mut a = $id::splat(false);
                a = a.replace(i, true);
                assert!(a.any());
            }
        }
        #[test]
        fn none() {
            use coresimd::simd::$id;

            let a = $id::splat(true);
            assert!(!a.none());
            let a = $id::splat(false);
            assert!(a.none());

            for i in 0..$id::lanes() {
                let mut a = $id::splat(true);
                a = a.replace(i, false);
                assert!(!a.none());
                let mut a = $id::splat(false);
                a = a.replace(i, true);
                assert!(!a.none());
            }
        }
    };
}
