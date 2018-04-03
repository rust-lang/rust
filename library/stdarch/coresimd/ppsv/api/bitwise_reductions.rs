//! Implements portable bitwise vector reductions.
#![allow(unused)]

macro_rules! impl_bitwise_reductions {
    ($id:ident, $elem_ty:ident) => {
        impl $id {
            /// Lane-wise bitwise `and` of the vector elements.
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn and(self) -> $elem_ty {
                use coresimd::simd_llvm::simd_reduce_and;
                unsafe { simd_reduce_and(self) }
            }
            /// Lane-wise bitwise `and` of the vector elements.
            #[cfg(target_arch = "aarch64")]
            #[inline]
            pub fn and(self) -> $elem_ty {
                // FIXME: broken on aarch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                let mut x = self.extract(0) as $elem_ty;
                for i in 1..$id::lanes() {
                    x &= self.extract(i) as $elem_ty;
                }
                x
            }

            /// Lane-wise bitwise `or` of the vector elements.
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn or(self) -> $elem_ty {
                use coresimd::simd_llvm::simd_reduce_or;
                unsafe { simd_reduce_or(self) }
            }
            /// Lane-wise bitwise `or` of the vector elements.
            #[cfg(target_arch = "aarch64")]
            #[inline]
            pub fn or(self) -> $elem_ty {
                // FIXME: broken on aarch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                let mut x = self.extract(0) as $elem_ty;
                for i in 1..$id::lanes() {
                    x |= self.extract(i) as $elem_ty;
                }
                x
            }

            /// Lane-wise bitwise `xor` of the vector elements.
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn xor(self) -> $elem_ty {
                use coresimd::simd_llvm::simd_reduce_xor;
                unsafe { simd_reduce_xor(self) }
            }
            /// Lane-wise bitwise `xor` of the vector elements.
            #[cfg(target_arch = "aarch64")]
            #[inline]
            pub fn xor(self) -> $elem_ty {
                // FIXME: broken on aarch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                let mut x = self.extract(0) as $elem_ty;
                for i in 1..$id::lanes() {
                    x ^= self.extract(i) as $elem_ty;
                }
                x
            }
        }
    };
}

macro_rules! impl_bool_bitwise_reductions {
    ($id:ident, $elem_ty:ident, $internal_ty:ident) => {
        impl $id {
            /// Lane-wise bitwise `and` of the vector elements.
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn and(self) -> $elem_ty {
                use coresimd::simd_llvm::simd_reduce_and;
                unsafe {
                    let r: $internal_ty = simd_reduce_and(self);
                    r != 0
                }
            }
            /// Lane-wise bitwise `and` of the vector elements.
            #[cfg(target_arch = "aarch64")]
            #[inline]
            pub fn and(self) -> $elem_ty {
                // FIXME: broken on aarch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                let mut x = self.extract(0) as $elem_ty;
                for i in 1..$id::lanes() {
                    x &= self.extract(i) as $elem_ty;
                }
                x
            }

            /// Lane-wise bitwise `or` of the vector elements.
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn or(self) -> $elem_ty {
                use coresimd::simd_llvm::simd_reduce_or;
                unsafe {
                    let r: $internal_ty = simd_reduce_or(self);
                    r != 0
                }
            }
            /// Lane-wise bitwise `or` of the vector elements.
            #[cfg(target_arch = "aarch64")]
            #[inline]
            pub fn or(self) -> $elem_ty {
                // FIXME: broken on aarch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                let mut x = self.extract(0) as $elem_ty;
                for i in 1..$id::lanes() {
                    x |= self.extract(i) as $elem_ty;
                }
                x
            }

            /// Lane-wise bitwise `xor` of the vector elements.
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn xor(self) -> $elem_ty {
                use coresimd::simd_llvm::simd_reduce_xor;
                unsafe {
                    let r: $internal_ty = simd_reduce_xor(self);
                    r != 0
                }
            }
            /// Lane-wise bitwise `xor` of the vector elements.
            #[cfg(target_arch = "aarch64")]
            #[inline]
            pub fn xor(self) -> $elem_ty {
                // FIXME: broken on aarch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                let mut x = self.extract(0) as $elem_ty;
                for i in 1..$id::lanes() {
                    x ^= self.extract(i) as $elem_ty;
                }
                x
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_bitwise_reductions {
    ($id:ident, $true:expr) => {
        #[test]
        fn and() {
            let false_ = !$true;
            use coresimd::simd::$id;
            let v = $id::splat(false_);
            assert_eq!(v.and(), false_);
            let v = $id::splat($true);
            assert_eq!(v.and(), $true);
            let v = $id::splat(false_);
            let v = v.replace(0, $true);
            assert_eq!(v.and(), false_);
            let v = $id::splat($true);
            let v = v.replace(0, false_);
            assert_eq!(v.and(), false_);
        }
        #[test]
        fn or() {
            let false_ = !$true;
            use coresimd::simd::$id;
            let v = $id::splat(false_);
            assert_eq!(v.or(), false_);
            let v = $id::splat($true);
            assert_eq!(v.or(), $true);
            let v = $id::splat(false_);
            let v = v.replace(0, $true);
            assert_eq!(v.or(), $true);
            let v = $id::splat($true);
            let v = v.replace(0, false_);
            assert_eq!(v.or(), $true);
        }
        #[test]
        fn xor() {
            let false_ = !$true;
            use coresimd::simd::$id;
            let v = $id::splat(false_);
            assert_eq!(v.xor(), false_);
            let v = $id::splat($true);
            assert_eq!(v.xor(), false_);
            let v = $id::splat(false_);
            let v = v.replace(0, $true);
            assert_eq!(v.xor(), $true);
            let v = $id::splat($true);
            let v = v.replace(0, false_);
            assert_eq!(v.xor(), $true);
        }
    };
}
