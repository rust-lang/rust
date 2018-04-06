//! Lane-wise bitwise operations for integer vectors and vector masks.
#![allow(unused)]

macro_rules! impl_bitwise_scalar_ops {
    ($id:ident, $elem_ty:ident) => {
        impl ::ops::BitXor<$elem_ty> for $id {
            type Output = Self;
            #[inline]
            fn bitxor(self, other: $elem_ty) -> Self {
                self ^ $id::splat(other)
            }
        }
        impl ::ops::BitXor<$id> for $elem_ty {
            type Output = $id;
            #[inline]
            fn bitxor(self, other: $id) -> $id {
                $id::splat(self) ^ other
            }
        }

        impl ::ops::BitAnd<$elem_ty> for $id {
            type Output = Self;
            #[inline]
            fn bitand(self, other: $elem_ty) -> Self {
                self & $id::splat(other)
            }
        }
        impl ::ops::BitAnd<$id> for $elem_ty {
            type Output = $id;
            #[inline]
            fn bitand(self, other: $id) -> $id {
                $id::splat(self) & other
            }
        }

        impl ::ops::BitOr<$elem_ty> for $id {
            type Output = Self;
            #[inline]
            fn bitor(self, other: $elem_ty) -> Self {
                self | $id::splat(other)
            }
        }
        impl ::ops::BitOr<$id> for $elem_ty {
            type Output = $id;
            #[inline]
            fn bitor(self, other: $id) -> $id {
                $id::splat(self) | other
            }
        }

        impl ::ops::BitAndAssign<$elem_ty> for $id {
            #[inline]
            fn bitand_assign(&mut self, other: $elem_ty) {
                *self = *self & other;
            }
        }
        impl ::ops::BitOrAssign<$elem_ty> for $id {
            #[inline]
            fn bitor_assign(&mut self, other: $elem_ty) {
                *self = *self | other;
            }
        }
        impl ::ops::BitXorAssign<$elem_ty> for $id {
            #[inline]
            fn bitxor_assign(&mut self, other: $elem_ty) {
                *self = *self ^ other;
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_int_bitwise_scalar_ops {
    ($id:ident, $elem_ty:ident) => {
        #[test]
        fn bitwise_scalar_ops() {
            use coresimd::simd::$id;
            let zi = 0 as $elem_ty;
            let oi = 1 as $elem_ty;
            let ti = 2 as $elem_ty;
            let z = $id::splat(zi);
            let o = $id::splat(oi);
            let t = $id::splat(ti);

            // BitAnd:
            assert_eq!(oi & o, o);
            assert_eq!(o & oi, o);
            assert_eq!(oi & z, z);
            assert_eq!(o & zi, z);
            assert_eq!(zi & o, z);
            assert_eq!(z & oi, z);
            assert_eq!(zi & z, z);
            assert_eq!(z & zi, z);

            assert_eq!(ti & t, t);
            assert_eq!(t & ti, t);
            assert_eq!(ti & o, z);
            assert_eq!(t & oi, z);
            assert_eq!(oi & t, z);
            assert_eq!(o & ti, z);

            // BitOr:
            assert_eq!(oi | o, o);
            assert_eq!(o | oi, o);
            assert_eq!(oi | z, o);
            assert_eq!(o | zi, o);
            assert_eq!(zi | o, o);
            assert_eq!(z | oi, o);
            assert_eq!(zi | z, z);
            assert_eq!(z | zi, z);

            assert_eq!(ti | t, t);
            assert_eq!(t | ti, t);
            assert_eq!(zi | t, t);
            assert_eq!(z | ti, t);
            assert_eq!(ti | z, t);
            assert_eq!(t | zi, t);

            // BitXOR:
            assert_eq!(oi ^ o, z);
            assert_eq!(o ^ oi, z);
            assert_eq!(zi ^ z, z);
            assert_eq!(z ^ zi, z);
            assert_eq!(zi ^ o, o);
            assert_eq!(z ^ oi, o);
            assert_eq!(oi ^ z, o);
            assert_eq!(o ^ zi, o);

            assert_eq!(ti ^ t, z);
            assert_eq!(t ^ ti, z);
            assert_eq!(ti ^ z, t);
            assert_eq!(t ^ zi, t);
            assert_eq!(zi ^ t, t);
            assert_eq!(z ^ ti, t);

            {
                // AndAssign:
                let mut v = o;
                v &= ti;
                assert_eq!(v, z);
            }
            {
                // OrAssign:
                let mut v = z;
                v |= oi;
                assert_eq!(v, o);
            }
            {
                // XORAssign:
                let mut v = z;
                v ^= oi;
                assert_eq!(v, o);
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_mask_bitwise_scalar_ops {
    ($id:ident) => {
        #[test]
        fn bool_scalar_arithmetic() {
            use coresimd::simd::*;

            let ti = true;
            let fi = false;
            let t = $id::splat(ti);
            let f = $id::splat(fi);
            assert!(t != f);
            assert!(!(t == f));

            // BitAnd:
            assert_eq!(ti & f, f);
            assert_eq!(t & fi, f);
            assert_eq!(fi & t, f);
            assert_eq!(f & ti, f);
            assert_eq!(ti & t, t);
            assert_eq!(t & ti, t);
            assert_eq!(fi & f, f);
            assert_eq!(f & fi, f);

            // BitOr:
            assert_eq!(ti | f, t);
            assert_eq!(t | fi, t);
            assert_eq!(fi | t, t);
            assert_eq!(f | ti, t);
            assert_eq!(ti | t, t);
            assert_eq!(t | ti, t);
            assert_eq!(fi | f, f);
            assert_eq!(f | fi, f);

            // BitXOR:
            assert_eq!(ti ^ f, t);
            assert_eq!(t ^ fi, t);
            assert_eq!(fi ^ t, t);
            assert_eq!(f ^ ti, t);
            assert_eq!(ti ^ t, f);
            assert_eq!(t ^ ti, f);
            assert_eq!(fi ^ f, f);
            assert_eq!(f ^ fi, f);

            {
                // AndAssign:
                let mut v = f;
                v &= ti;
                assert_eq!(v, f);
            }
            {
                // OrAssign:
                let mut v = f;
                v |= ti;
                assert_eq!(v, t);
            }
            {
                // XORAssign:
                let mut v = f;
                v ^= ti;
                assert_eq!(v, t);
            }
        }
    };
}
