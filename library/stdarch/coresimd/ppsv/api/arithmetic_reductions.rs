//! Implements portable arithmetic vector reductions.

macro_rules! impl_arithmetic_reductions {
    ($id:ident, $elem_ty:ident) => {
        impl $id {
            /// Lane-wise addition of the vector elements.
            #[inline]
            pub fn sum(self) -> $elem_ty {
                ReduceAdd::reduce_add(self)
            }
            /// Lane-wise multiplication of the vector elements.
            #[inline]
            pub fn product(self) -> $elem_ty {
                ReduceMul::reduce_mul(self)
            }
        }
    }
}

#[cfg(test)]
macro_rules! test_arithmetic_reductions {
    ($id:ident, $elem_ty:ident) => {

        fn alternating(x: usize) -> ::coresimd::simd::$id {
            use ::coresimd::simd::$id;
            let mut v = $id::splat(1 as $elem_ty);
            for i in 0..$id::lanes() {
                if i % x == 0 {
                    v = v.replace(i, 2 as $elem_ty);
                }
            }
            v
        }

        #[test]
        fn sum() {
            use ::coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.sum(), 0 as $elem_ty);
            let v = $id::splat(1 as $elem_ty);
            assert_eq!(v.sum(), $id::lanes() as $elem_ty);
            let v = alternating(2);
            eprintln!("{:?}", v);
            assert_eq!(v.sum(), ($id::lanes() / 2 + $id::lanes()) as $elem_ty);
        }
        #[test]
        fn product() {
            use ::coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.product(), 0 as $elem_ty);
            let v = $id::splat(1 as $elem_ty);
            assert_eq!(v.product(), 1 as $elem_ty);
            let f = match $id::lanes() {
                64 => 16,
                32 => 8,
                16 => 4,
                _ => 2,
            };
            let v = alternating(f);
            eprintln!("{:?}", v);
            assert_eq!(v.product(), (2_usize.pow(($id::lanes() / f) as u32) as $elem_ty));
        }
    }
}
