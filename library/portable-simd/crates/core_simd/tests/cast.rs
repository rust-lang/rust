#![feature(portable_simd)]
macro_rules! cast_types {
    ($start:ident, $($target:ident),*) => {
        mod $start {
            #[allow(unused)]
            use core_simd::simd::prelude::*;
            type Vector<const N: usize> = Simd<$start, N>;
            $(
                mod $target {
                    use super::*;
                    test_helpers::test_lanes! {
                        fn cast_as<const N: usize>() {
                            test_helpers::test_unary_elementwise(
                                &Vector::<N>::cast::<$target>,
                                &|x| x as $target,
                                &|_| true,
                            )
                        }
                    }
                }
            )*
        }
    };
}

// The hypothesis is that widening conversions aren't terribly interesting.
cast_types!(f32, f64, i8, u8, usize, isize);
cast_types!(f64, f32, i8, u8, usize, isize);
cast_types!(i8, u8, f32);
cast_types!(u8, i8, f32);
cast_types!(i16, u16, i8, u8, f32);
cast_types!(u16, i16, i8, u8, f32);
cast_types!(i32, u32, i8, u8, f32, f64);
cast_types!(u32, i32, i8, u8, f32, f64);
cast_types!(i64, u64, i8, u8, isize, usize, f32, f64);
cast_types!(u64, i64, i8, u8, isize, usize, f32, f64);
cast_types!(isize, usize, i8, u8, f32, f64);
cast_types!(usize, isize, i8, u8, f32, f64);
