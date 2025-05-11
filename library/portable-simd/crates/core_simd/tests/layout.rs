#![feature(portable_simd)]

macro_rules! layout_tests {
    { $($mod:ident, $ty:ty,)* } => {
        $(
        mod $mod {
            test_helpers::test_lanes! {
                fn no_padding<const LANES: usize>() {
                    assert_eq!(
                        size_of::<core_simd::simd::Simd::<$ty, LANES>>(),
                        size_of::<[$ty; LANES]>(),
                    );
                }
            }
        }
        )*
    }
}

layout_tests! {
    i8, i8,
    i16, i16,
    i32, i32,
    i64, i64,
    isize, isize,
    u8, u8,
    u16, u16,
    u32, u32,
    u64, u64,
    usize, usize,
    f32, f32,
    f64, f64,
    mut_ptr, *mut (),
    const_ptr, *const (),
}
