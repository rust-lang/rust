pub(crate) trait BitEq {
    fn biteq(&self, other: &Self) -> bool;
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result;
}

macro_rules! impl_biteq {
    { integer impl BitEq for $($type:ty,)* } => {
        $(
        impl BitEq for $type {
            fn biteq(&self, other: &Self) -> bool {
                self == other
            }

            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "{:?} ({:x})", self, self)
            }
        }
        )*
    };
    { float impl BitEq for $($type:ty,)* } => {
        $(
        impl BitEq for $type {
            fn biteq(&self, other: &Self) -> bool {
                self.to_bits() == other.to_bits()
            }

            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "{:?} ({:x})", self, self.to_bits())
            }
        }
        )*
    };
    { vector impl BitEq for $($type:ty,)* } => {
        $(
        impl BitEq for $type {
            fn biteq(&self, other: &Self) -> bool {
                let a: &[_] = self.as_ref();
                let b: &[_] = other.as_ref();
                if a.len() == b.len() {
                    a.iter().zip(b.iter()).fold(true, |value, (left, right)| {
                        value && left.biteq(right)
                    })
                } else {
                    false
                }
            }

            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                #[repr(transparent)]
                struct Wrapper<'a, T: BitEq>(&'a T);

                impl<T: BitEq> core::fmt::Debug for Wrapper<'_, T> {
                    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                        self.0.fmt(f)
                    }
                }

                let slice: &[_] = self.as_ref();

                f.debug_list()
                    .entries(slice.iter().map(|x| Wrapper(x)))
                    .finish()
            }
        }
        )*
    };
}

impl_biteq! {
    integer impl BitEq for
        u8, u16, u32, u64, u128, usize,
        i8, i16, i32, i64, i128, isize,
}

impl_biteq! {
    float impl BitEq for f32, f64,
}

impl_biteq! {
    vector impl BitEq for
        core_simd::u8x8,    core_simd::u8x16,   core_simd::u8x32,   core_simd::u8x64,
        core_simd::i8x8,    core_simd::i8x16,   core_simd::i8x32,   core_simd::i8x64,
        core_simd::u16x4,   core_simd::u16x8,   core_simd::u16x16,  core_simd::u16x32,
        core_simd::i16x4,   core_simd::i16x8,   core_simd::i16x16,  core_simd::i16x32,
        core_simd::u32x2,   core_simd::u32x4,   core_simd::u32x8,   core_simd::u32x16,
        core_simd::i32x2,   core_simd::i32x4,   core_simd::i32x8,   core_simd::i32x16,
        core_simd::u64x2,   core_simd::u64x4,   core_simd::u64x8,
        core_simd::i64x2,   core_simd::i64x4,   core_simd::i64x8,
        core_simd::u128x2,  core_simd::u128x4,
        core_simd::i128x2,  core_simd::i128x4,
        core_simd::usizex2, core_simd::usizex4, core_simd::usizex8,
        core_simd::isizex2, core_simd::isizex4, core_simd::isizex8,
        core_simd::f32x2, core_simd::f32x4, core_simd::f32x8, core_simd::f32x16,
        core_simd::f64x2, core_simd::f64x4, core_simd::f64x8,
}

pub(crate) struct BitEqWrapper<'a, T>(pub(crate) &'a T);

impl<T: BitEq> PartialEq for BitEqWrapper<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.biteq(other.0)
    }
}

impl<T: BitEq> core::fmt::Debug for BitEqWrapper<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

macro_rules! assert_biteq {
    { $a:expr, $b:expr } => {
        {
            use helpers::biteq::BitEqWrapper;
            let a = $a;
            let b = $b;
            assert_eq!(BitEqWrapper(&a), BitEqWrapper(&b));
        }
    }
}
