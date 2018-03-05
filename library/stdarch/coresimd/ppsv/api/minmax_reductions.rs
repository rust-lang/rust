//! Implements portable arithmetic vector reductions.

macro_rules! impl_minmax_reductions {
    ($id:ident, $elem_ty:ident) => {
        impl $id {
            /// Largest vector value.
            #[inline]
            pub fn max(self) -> $elem_ty {
                ReduceMax::reduce_max(self)
            }
            /// Smallest vector value.
            #[inline]
            pub fn min(self) -> $elem_ty {
                ReduceMin::reduce_min(self)
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
