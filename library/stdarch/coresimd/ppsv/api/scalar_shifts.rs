//! Implements integer shifts.
#![allow(unused)]

macro_rules! impl_shifts {
    ($id:ident, $elem_ty:ident, $($by:ident),+) => {
        $(
            impl ::ops::Shl<$by> for $id {
                type Output = Self;
                #[inline]
                fn shl(self, other: $by) -> Self {
                    use coresimd::simd_llvm::simd_shl;
                    unsafe { simd_shl(self, $id::splat(other as $elem_ty)) }
                }
            }
            impl ::ops::Shr<$by> for $id {
                type Output = Self;
                #[inline]
                fn shr(self, other: $by) -> Self {
                    use coresimd::simd_llvm::simd_shr;
                    unsafe { simd_shr(self, $id::splat(other as $elem_ty)) }
                }
            }

            impl ::ops::ShlAssign<$by> for $id {
                #[inline]
                fn shl_assign(&mut self, other: $by) {
                    *self = *self << other;
                }
            }
            impl ::ops::ShrAssign<$by> for $id {
                #[inline]
                fn shr_assign(&mut self, other: $by) {
                    *self = *self >> other;
                }
            }

        )+
    }
}

macro_rules! impl_all_scalar_shifts {
    ($id:ident, $elem_ty:ident) => {
        impl_shifts!(
            $id, $elem_ty,
            u8, u16, u32, u64, usize,
            i8, i16, i32, i64, isize);

    }
}

#[cfg(test)]
macro_rules! test_shift_ops {
    ($id:ident, $elem_ty:ident, $($index_ty:ident),+) => {
        #[test]
        fn scalar_shift_ops() {
            use ::coresimd::simd::$id;
            use ::std::mem;
            let z = $id::splat(0 as $elem_ty);
            let o = $id::splat(1 as $elem_ty);
            let t = $id::splat(2 as $elem_ty);
            let f = $id::splat(4 as $elem_ty);

            $(
                {
                    let zi = 0 as $index_ty;
                    let oi = 1 as $index_ty;
                    let ti = 2 as $index_ty;
                    let maxi = (mem::size_of::<$elem_ty>() * 8 - 1) as $index_ty;

                    // shr
                    assert_eq!(z >> zi, z);
                    assert_eq!(z >> oi, z);
                    assert_eq!(z >> ti, z);
                    assert_eq!(z >> ti, z);

                    assert_eq!(o >> zi, o);
                    assert_eq!(t >> zi, t);
                    assert_eq!(f >> zi, f);
                    assert_eq!(f >> maxi, z);

                    assert_eq!(o >> oi, z);
                    assert_eq!(t >> oi, o);
                    assert_eq!(t >> ti, z);
                    assert_eq!(f >> oi, t);
                    assert_eq!(f >> ti, o);
                    assert_eq!(f >> maxi, z);

                    // shl
                    assert_eq!(z << zi, z);
                    assert_eq!(o << zi, o);
                    assert_eq!(t << zi, t);
                    assert_eq!(f << zi, f);
                    assert_eq!(f << maxi, z);

                    assert_eq!(o << oi, t);
                    assert_eq!(o << ti, f);
                    assert_eq!(t << oi, f);

                    {  // shr_assign
                        let mut v = o;
                        v >>= oi;
                        assert_eq!(v, z);
                    }
                    {  // shl_assign
                        let mut v = o;
                        v <<= oi;
                        assert_eq!(v, t);
                    }
                }
            )+
        }
    };
}

#[cfg(test)]
macro_rules! test_all_scalar_shift_ops {
    ($id:ident, $elem_ty:ident) => {
        test_shift_ops!(
            $id, $elem_ty,
            u8, u16, u32, u64, usize,
            i8, i16, i32, i64, isize);
    }
}
