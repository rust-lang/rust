//! Implements integer shifts.
#![allow(unused)]

macro_rules! impl_vector_shifts {
    ($id:ident, $elem_ty:ident) => {
        impl ::ops::Shl<$id> for $id {
            type Output = Self;
            #[inline]
            fn shl(self, other: Self) -> Self {
                use coresimd::simd_llvm::simd_shl;
                unsafe { simd_shl(self, other) }
            }
        }
        impl ::ops::Shr<$id> for $id {
            type Output = Self;
            #[inline]
            fn shr(self, other: Self) -> Self {
                use coresimd::simd_llvm::simd_shr;
                unsafe { simd_shr(self, other) }
            }
        }
        impl ::ops::ShlAssign<$id> for $id {
            #[inline]
            fn shl_assign(&mut self, other: Self) {
                *self = *self << other;
            }
        }
        impl ::ops::ShrAssign<$id> for $id {
            #[inline]
            fn shr_assign(&mut self, other: Self) {
                *self = *self >> other;
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_vector_shift_ops {
    ($id:ident, $elem_ty:ident) => {
        #[test]
        fn shift_ops() {
            use coresimd::simd::$id;
            use std::mem;
            let z = $id::splat(0 as $elem_ty);
            let o = $id::splat(1 as $elem_ty);
            let t = $id::splat(2 as $elem_ty);
            let f = $id::splat(4 as $elem_ty);

            let max =
                $id::splat((mem::size_of::<$elem_ty>() * 8 - 1) as $elem_ty);

            // shr
            assert_eq!(z >> z, z);
            assert_eq!(z >> o, z);
            assert_eq!(z >> t, z);
            assert_eq!(z >> t, z);

            assert_eq!(o >> z, o);
            assert_eq!(t >> z, t);
            assert_eq!(f >> z, f);
            assert_eq!(f >> max, z);

            assert_eq!(o >> o, z);
            assert_eq!(t >> o, o);
            assert_eq!(t >> t, z);
            assert_eq!(f >> o, t);
            assert_eq!(f >> t, o);
            assert_eq!(f >> max, z);

            // shl
            assert_eq!(z << z, z);
            assert_eq!(o << z, o);
            assert_eq!(t << z, t);
            assert_eq!(f << z, f);
            assert_eq!(f << max, z);

            assert_eq!(o << o, t);
            assert_eq!(o << t, f);
            assert_eq!(t << o, f);

            {
                // shr_assign
                let mut v = o;
                v >>= o;
                assert_eq!(v, z);
            }
            {
                // shl_assign
                let mut v = o;
                v <<= o;
                assert_eq!(v, t);
            }
        }
    };
}
