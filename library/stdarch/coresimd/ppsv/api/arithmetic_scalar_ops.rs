//! Lane-wise arithmetic operations.
#![allow(unused)]

macro_rules! impl_arithmetic_scalar_ops {
    ($id: ident, $elem_ty: ident) => {
        impl ::ops::Add<$elem_ty> for $id {
            type Output = Self;
            #[inline]
            fn add(self, other: $elem_ty) -> Self {
                self + $id::splat(other)
            }
        }
        impl ::ops::Add<$id> for $elem_ty {
            type Output = $id;
            #[inline]
            fn add(self, other: $id) -> $id {
                $id::splat(self) + other
            }
        }

        impl ::ops::Sub<$elem_ty> for $id {
            type Output = Self;
            #[inline]
            fn sub(self, other: $elem_ty) -> Self {
                self - $id::splat(other)
            }
        }
        impl ::ops::Sub<$id> for $elem_ty {
            type Output = $id;
            #[inline]
            fn sub(self, other: $id) -> $id {
                $id::splat(self) - other
            }
        }

        impl ::ops::Mul<$elem_ty> for $id {
            type Output = Self;
            #[inline]
            fn mul(self, other: $elem_ty) -> Self {
                self * $id::splat(other)
            }
        }
        impl ::ops::Mul<$id> for $elem_ty {
            type Output = $id;
            #[inline]
            fn mul(self, other: $id) -> $id {
                $id::splat(self) * other
            }
        }

        impl ::ops::Div<$elem_ty> for $id {
            type Output = Self;
            #[inline]
            fn div(self, other: $elem_ty) -> Self {
                self / $id::splat(other)
            }
        }
        impl ::ops::Div<$id> for $elem_ty {
            type Output = $id;
            #[inline]
            fn div(self, other: $id) -> $id {
                $id::splat(self) / other
            }
        }

        impl ::ops::Rem<$elem_ty> for $id {
            type Output = Self;
            #[inline]
            fn rem(self, other: $elem_ty) -> Self {
                self % $id::splat(other)
            }
        }
        impl ::ops::Rem<$id> for $elem_ty {
            type Output = $id;
            #[inline]
            fn rem(self, other: $id) -> $id {
                $id::splat(self) % other
            }
        }

        impl ::ops::AddAssign<$elem_ty> for $id {
            #[inline]
            fn add_assign(&mut self, other: $elem_ty) {
                *self = *self + other;
            }
        }

        impl ::ops::SubAssign<$elem_ty> for $id {
            #[inline]
            fn sub_assign(&mut self, other: $elem_ty) {
                *self = *self - other;
            }
        }

        impl ::ops::MulAssign<$elem_ty> for $id {
            #[inline]
            fn mul_assign(&mut self, other: $elem_ty) {
                *self = *self * other;
            }
        }

        impl ::ops::DivAssign<$elem_ty> for $id {
            #[inline]
            fn div_assign(&mut self, other: $elem_ty) {
                *self = *self / other;
            }
        }

        impl ::ops::RemAssign<$elem_ty> for $id {
            #[inline]
            fn rem_assign(&mut self, other: $elem_ty) {
                *self = *self % other;
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_arithmetic_scalar_ops {
    ($id: ident, $elem_ty: ident) => {
        #[test]
        fn arithmetic_scalar() {
            use coresimd::simd::$id;
            let zi = 0 as $elem_ty;
            let oi = 1 as $elem_ty;
            let ti = 2 as $elem_ty;
            let fi = 4 as $elem_ty;
            let z = $id::splat(zi);
            let o = $id::splat(oi);
            let t = $id::splat(ti);
            let f = $id::splat(fi);

            // add
            assert_eq!(zi + z, z);
            assert_eq!(z + zi, z);
            assert_eq!(oi + z, o);
            assert_eq!(o + zi, o);
            assert_eq!(ti + z, t);
            assert_eq!(t + zi, t);
            assert_eq!(ti + t, f);
            assert_eq!(t + ti, f);
            // sub
            assert_eq!(zi - z, z);
            assert_eq!(z - zi, z);
            assert_eq!(oi - z, o);
            assert_eq!(o - zi, o);
            assert_eq!(ti - z, t);
            assert_eq!(t - zi, t);
            assert_eq!(fi - t, t);
            assert_eq!(f - ti, t);
            assert_eq!(f - o - o, t);
            assert_eq!(f - oi - oi, t);
            // mul
            assert_eq!(zi * z, z);
            assert_eq!(z * zi, z);
            assert_eq!(zi * o, z);
            assert_eq!(z * oi, z);
            assert_eq!(zi * t, z);
            assert_eq!(z * ti, z);
            assert_eq!(oi * t, t);
            assert_eq!(o * ti, t);
            assert_eq!(ti * t, f);
            assert_eq!(t * ti, f);
            // div
            assert_eq!(zi / o, z);
            assert_eq!(z / oi, z);
            assert_eq!(ti / o, t);
            assert_eq!(t / oi, t);
            assert_eq!(fi / o, f);
            assert_eq!(f / oi, f);
            assert_eq!(ti / t, o);
            assert_eq!(t / ti, o);
            assert_eq!(fi / t, t);
            assert_eq!(f / ti, t);
            // rem
            assert_eq!(oi % o, z);
            assert_eq!(o % oi, z);
            assert_eq!(fi % t, z);
            assert_eq!(f % ti, z);

            {
                let mut v = z;
                assert_eq!(v, z);
                v += oi; // add_assign
                assert_eq!(v, o);
                v -= oi; // sub_assign
                assert_eq!(v, z);
                v = t;
                v *= oi; // mul_assign
                assert_eq!(v, t);
                v *= ti;
                assert_eq!(v, f);
                v /= oi; // div_assign
                assert_eq!(v, f);
                v /= ti;
                assert_eq!(v, t);
                v %= ti; // rem_assign
                assert_eq!(v, z);
            }
        }
    };
}
