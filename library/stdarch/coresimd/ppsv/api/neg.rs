//! Implements `std::ops::Neg` for signed vector types.

macro_rules! impl_neg_op {
    ($id:ident, $elem_ty:ident) => {
        impl ops::Neg for $id {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self::splat(-1 as $elem_ty) * self
            }
        }
    };
}

#[cfg(test)]
#[macro_export]
macro_rules! test_neg_op {
    ($id:ident, $elem_ty:ident) => {
        #[test]
        fn neg() {
            use ::coresimd::simd::$id;
            let z = $id::splat(0 as $elem_ty);
            let o = $id::splat(1 as $elem_ty);
            let t = $id::splat(2 as $elem_ty);
            let f = $id::splat(4 as $elem_ty);

            let nz = $id::splat(-(0 as $elem_ty));
            let no = $id::splat(-(1 as $elem_ty));
            let nt = $id::splat(-(2 as $elem_ty));
            let nf = $id::splat(-(4 as $elem_ty));

            assert_eq!(-z, nz);
            assert_eq!(-o, no);
            assert_eq!(-t, nt);
            assert_eq!(-f, nf);

            assert_eq!(z, -nz);
            assert_eq!(o, -no);
            assert_eq!(t, -nt);
            assert_eq!(f, -nf);
        }
    };
}
