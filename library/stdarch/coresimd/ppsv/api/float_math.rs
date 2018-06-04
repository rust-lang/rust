//! Float math

macro_rules! impl_float_math {
    ($id:ident) => {
        impl $id {
            /// Absolute-value
            #[inline]
            pub fn abs(self) -> Self {
                use coresimd::ppsv::codegen::abs::FloatAbs;
                FloatAbs::abs(self)
            }

            /// Square-root
            #[inline]
            pub fn sqrt(self) -> Self {
                use coresimd::ppsv::codegen::sqrt::FloatSqrt;
                FloatSqrt::sqrt(self)
            }

            /// Square-root estimate
            #[inline]
            pub fn sqrte(self) -> Self {
                use coresimd::simd_llvm::simd_fsqrt;
                unsafe { simd_fsqrt(self) }
            }

            /// Reciprocal square-root estimate
            #[inline]
            pub fn rsqrte(self) -> Self {
                unsafe {
                    use coresimd::simd_llvm::simd_fsqrt;
                    $id::splat(1.) / simd_fsqrt(self)
                }
            }

            /// Fused multiply add: `self * y + z`
            #[inline]
            pub fn fma(self, y: Self, z: Self) -> Self {
                use coresimd::ppsv::codegen::fma::FloatFma;
                FloatFma::fma(self, y, z)
            }

            /// Sin
            #[inline(always)]
            pub fn sin(self) -> Self {
                use coresimd::ppsv::codegen::sin::FloatSin;
                FloatSin::sin(self)
            }

            /// Cos
            #[inline]
            pub fn cos(self) -> Self {
                use coresimd::ppsv::codegen::cos::FloatCos;
                FloatCos::cos(self)
            }
        }
    };
}

macro_rules! test_float_math {
    ($id:ident, $elem_ty:ident) => {

        fn sqrt2() -> $elem_ty {
            match ::mem::size_of::<$elem_ty>() {
                4 => 1.4142135 as $elem_ty,
                8 => 1.4142135623730951 as $elem_ty,
                _ => unreachable!(),
            }
        }

        fn pi() -> $elem_ty {
            match ::mem::size_of::<$elem_ty>() {
                4 => ::std::f32::consts::PI as $elem_ty,
                8 => ::std::f64::consts::PI as $elem_ty,
                _ => unreachable!(),
            }
        }

        #[test]
        fn abs() {
            use coresimd::simd::*;
            let o = $id::splat(1 as $elem_ty);
            assert_eq!(o, o.abs());

            let mo = $id::splat(-1 as $elem_ty);
            assert_eq!(o, mo.abs());
        }

        #[test]
        fn sqrt() {
            use coresimd::simd::*;
            let z = $id::splat(0 as $elem_ty);
            let o = $id::splat(1 as $elem_ty);
            assert_eq!(z, z.sqrt());
            assert_eq!(o, o.sqrt());

            let t = $id::splat(2 as $elem_ty);
            let e = $id::splat(sqrt2() as $elem_ty);
            assert_eq!(e, t.sqrt());
        }

        #[test]
        fn sqrte() {
            use coresimd::simd::*;
            let z = $id::splat(0 as $elem_ty);
            let o = $id::splat(1 as $elem_ty);
            assert_eq!(z, z.sqrte());
            assert_eq!(o, o.sqrte());

            let t = $id::splat(2 as $elem_ty);
            let e = $id::splat(sqrt2() as $elem_ty);
            let error = (e - t.sqrte()).abs();
            let tol = $id::splat(2.4e-4 as $elem_ty);

            assert!(error.le(tol).all());
        }

        #[test]
        fn rsqrte() {
            use coresimd::simd::*;
            let o = $id::splat(1 as $elem_ty);
            assert_eq!(o, o.rsqrte());

            let t = $id::splat(2 as $elem_ty);
            let e = 1. / sqrt2();
            let error = (e - t.rsqrte()).abs();
            let tol = $id::splat(2.4e-4 as $elem_ty);
            assert!(error.le(tol).all());
        }

        #[test]
        fn fma() {
            use coresimd::simd::*;
            let z = $id::splat(0 as $elem_ty);
            let o = $id::splat(1 as $elem_ty);
            let t = $id::splat(2 as $elem_ty);
            let t3 = $id::splat(3 as $elem_ty);
            let f = $id::splat(4 as $elem_ty);

            assert_eq!(z, z.fma(z, z));
            assert_eq!(o, o.fma(o, z));
            assert_eq!(o, o.fma(z, o));
            assert_eq!(o, z.fma(o, o));

            assert_eq!(t, o.fma(o, o));
            assert_eq!(t, o.fma(t, z));
            assert_eq!(t, t.fma(o, z));

            assert_eq!(f, t.fma(t, z));
            assert_eq!(f, t.fma(o, t));
            assert_eq!(t3, t.fma(o, o));
        }

        #[test]
        fn sin() {
            use coresimd::simd::*;
            let z = $id::splat(0 as $elem_ty);
            let p = $id::splat(pi() as $elem_ty);
            let ph = $id::splat(pi() as $elem_ty / 2.);
            let o_r = $id::splat((pi() as $elem_ty / 2.).sin());
            let z_r = $id::splat((pi() as $elem_ty).sin());

            assert_eq!(z, z.sin());
            assert_eq!(o_r, ph.sin());
            assert_eq!(z_r, p.sin());
        }

        #[test]
        fn cos() {
            use coresimd::simd::*;
            let z = $id::splat(0 as $elem_ty);
            let o = $id::splat(1 as $elem_ty);
            let p = $id::splat(pi() as $elem_ty);
            let ph = $id::splat(pi() as $elem_ty / 2.);
            let z_r = $id::splat((pi() as $elem_ty / 2.).cos());
            let o_r = $id::splat((pi() as $elem_ty).cos());

            assert_eq!(o, z.cos());
            assert_eq!(z_r, ph.cos());
            assert_eq!(o_r, p.cos());
        }
    };
}
