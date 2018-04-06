//! Lane-wise arithmetic operations.
#![allow(unused)]

macro_rules! impl_int_minmax_ops {
    ($id:ident) => {
        impl $id {
            // Note:
            //
            // * if two elements are equal min returns
            //   always the second element
            // * if two elements are equal max returns
            //   always the second element
            //
            // Since we are dealing with integers here, and `min` and `max`
            // construct a new integer vector, whether the first or the
            // second element is returned when two elements compare equal
            // does not matter.

            /// Minimum of two vectors.
            ///
            /// Returns a new vector containing the minimum value of each of
            /// the input vector lanes.
            #[inline]
            pub fn min(self, x: Self) -> Self {
                self.lt(x).select(self, x)
            }

            /// Maximum of two vectors.
            ///
            /// Returns a new vector containing the minimum value of each of
            /// the input vector lanes.
            #[inline]
            pub fn max(self, x: Self) -> Self {
                self.gt(x).select(self, x)
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_int_minmax_ops {
    ($id:ident, $elem_ty:ident) => {
        #[test]
        fn minmax() {
            use coresimd::simd::$id;
            let o = $id::splat(1 as $elem_ty);
            let t = $id::splat(2 as $elem_ty);

            let mut m = o;
            for i in 0..$id::lanes() {
                if i % 2 == 0 {
                    m = m.replace(i, 2 as $elem_ty);
                }
            }

            assert_eq!(o.min(t), o);
            assert_eq!(t.min(o), o);
            assert_eq!(m.min(o), o);
            assert_eq!(o.min(m), o);
            assert_eq!(m.min(t), m);
            assert_eq!(t.min(m), m);

            assert_eq!(o.max(t), t);
            assert_eq!(t.max(o), t);
            assert_eq!(m.max(o), m);
            assert_eq!(o.max(m), m);
            assert_eq!(m.max(t), t);
            assert_eq!(t.max(m), t);
        }
    };
}

macro_rules! impl_float_minmax_ops {
    ($id:ident) => {
        impl $id {
            /// Minimum of two vectors.
            ///
            /// Returns a new vector containing the minimum value of each of the
            /// input vector lanes. The lane-wise semantics are the same as that
            /// of `min` for the primitive floating-point types.
            #[inline]
            pub fn min(self, x: Self) -> Self {
                use coresimd::simd_llvm::simd_fmin;
                unsafe { simd_fmin(self, x) }
            }

            /// Maximum of two vectors.
            ///
            /// Returns a new vector containing the minimum value of each of the
            /// input vector lanes. The lane-wise semantics are the same as that
            /// of `max` for the primitive floating-point types.
            #[inline]
            pub fn max(self, x: Self) -> Self {
                // FIXME: https://github.com/rust-lang-nursery/stdsimd/issues/416
                // use coresimd::simd_llvm::simd_fmax;
                // unsafe { simd_fmax(self, x) }
                use num::Float;
                let mut r = self;
                for i in 0..$id::lanes() {
                    let a = self.extract(i);
                    let b = x.extract(i);
                    r = r.replace(i, a.max(b))
                }
                r
            }
        }
    }
}

#[cfg(test)]
macro_rules! test_float_minmax_ops {
    ($id:ident, $elem_ty:ident) => {
        #[test]
        fn minmax() {
            use coresimd::simd::$id;
            let n = ::std::$elem_ty::NAN;
            let o = $id::splat(1. as $elem_ty);
            let t = $id::splat(2. as $elem_ty);

            let mut m = o;
            let mut on = o;
            for i in 0..$id::lanes() {
                if i % 2 == 0 {
                    m = m.replace(i, 2. as $elem_ty);
                    on = on.replace(i, n);
                }
            }

            assert_eq!(o.min(t), o);
            assert_eq!(t.min(o), o);
            assert_eq!(m.min(o), o);
            assert_eq!(o.min(m), o);
            assert_eq!(m.min(t), m);
            assert_eq!(t.min(m), m);

            assert_eq!(o.max(t), t);
            assert_eq!(t.max(o), t);
            assert_eq!(m.max(o), m);
            assert_eq!(o.max(m), m);
            assert_eq!(m.max(t), t);
            assert_eq!(t.max(m), t);

            assert_eq!(on.min(o), o);
            assert_eq!(o.min(on), o);
            assert_eq!(on.max(o), o);
            assert_eq!(o.max(on), o);
        }
    };
}
