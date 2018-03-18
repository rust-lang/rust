//! Lane-wise bitwise operations for integer and boolean vectors.
#![allow(unused)]

macro_rules! impl_bitwise_ops {
    ($ty:ident, $true_val:expr) => {
        impl ::ops::Not for $ty {
            type Output = Self;
            #[inline]
            fn not(self) -> Self {
                Self::splat($true_val) ^ self
            }
        }
        impl ::ops::BitXor for $ty {
            type Output = Self;
            #[inline]
            fn bitxor(self, other: Self) -> Self {
                use coresimd::simd_llvm::simd_xor;
                unsafe { simd_xor(self, other) }
            }
        }
        impl ::ops::BitAnd for $ty {
            type Output = Self;
            #[inline]
            fn bitand(self, other: Self) -> Self {
                use coresimd::simd_llvm::simd_and;
                unsafe { simd_and(self, other) }
            }
        }
        impl ::ops::BitOr for $ty {
            type Output = Self;
            #[inline]
            fn bitor(self, other: Self) -> Self {
                use coresimd::simd_llvm::simd_or;
                unsafe { simd_or(self, other) }
            }
        }
        impl ::ops::BitAndAssign for $ty {
            #[inline]
            fn bitand_assign(&mut self, other: Self) {
                *self = *self & other;
            }
        }
        impl ::ops::BitOrAssign for $ty {
            #[inline]
            fn bitor_assign(&mut self, other: Self) {
                *self = *self | other;
            }
        }
        impl ::ops::BitXorAssign for $ty {
            #[inline]
            fn bitxor_assign(&mut self, other: Self) {
                *self = *self ^ other;
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_int_bitwise_ops {
    ($id:ident, $elem_ty:ident) => {
        #[test]
        fn bitwise_ops() {
            use ::coresimd::simd::$id;
            let z = $id::splat(0 as $elem_ty);
            let o = $id::splat(1 as $elem_ty);
            let t = $id::splat(2 as $elem_ty);
            let m = $id::splat(!z.extract(0));

            // Not:
            assert_eq!(!z, m);
            assert_eq!(!m, z);

            // BitAnd:
            assert_eq!(o & o, o);
            assert_eq!(o & z, z);
            assert_eq!(z & o, z);
            assert_eq!(z & z, z);

            assert_eq!(t & t, t);
            assert_eq!(t & o, z);
            assert_eq!(o & t, z);

            // BitOr:
            assert_eq!(o | o, o);
            assert_eq!(o | z, o);
            assert_eq!(z | o, o);
            assert_eq!(z | z, z);

            assert_eq!(t | t, t);
            assert_eq!(z | t, t);
            assert_eq!(t | z, t);

            // BitXOR:
            assert_eq!(o ^ o, z);
            assert_eq!(z ^ z, z);
            assert_eq!(z ^ o, o);
            assert_eq!(o ^ z, o);

            assert_eq!(t ^ t, z);
            assert_eq!(t ^ z, t);
            assert_eq!(z ^ t, t);

            {  // AndAssign:
                let mut v = o;
                v &= t;
                assert_eq!(v, z);
            }
            {  // OrAssign:
                let mut v = z;
                v |= o;
                assert_eq!(v, o);
            }
            {  // XORAssign:
                let mut v = z;
                v ^= o;
                assert_eq!(v, o);
            }
        }
    }
}

#[cfg(test)]
macro_rules! test_bool_bitwise_ops {
    ($id:ident) => {
        #[test]
        fn bool_arithmetic() {
            use ::coresimd::simd::*;

            let t = $id::splat(true);
            let f = $id::splat(false);
            assert!(t != f);
            assert!(!(t == f));

            // Not:
            assert_eq!(!t, f);
            assert_eq!(t, !f);

            // BitAnd:
            assert_eq!(t & f, f);
            assert_eq!(f & t, f);
            assert_eq!(t & t, t);
            assert_eq!(f & f, f);

            // BitOr:
            assert_eq!(t | f, t);
            assert_eq!(f | t, t);
            assert_eq!(t | t, t);
            assert_eq!(f | f, f);

            // BitXOR:
            assert_eq!(t ^ f, t);
            assert_eq!(f ^ t, t);
            assert_eq!(t ^ t, f);
            assert_eq!(f ^ f, f);

            {  // AndAssign:
                let mut v = f;
                v &= t;
                assert_eq!(v, f);
            }
            {  // OrAssign:
                let mut v = f;
                v |= t;
                assert_eq!(v, t);
            }
            {  // XORAssign:
                let mut v = f;
                v ^= t;
                assert_eq!(v, t);
            }
        }
    }
}
