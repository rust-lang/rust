/// Mask-related operations using a particular mask layout.
pub trait MaskExt<Mask> {
    /// Test if each lane is equal to the corresponding lane in `other`.
    fn lanes_eq(self, other: Self) -> Mask;

    /// Test if each lane is not equal to the corresponding lane in `other`.
    fn lanes_ne(self, other: Self) -> Mask;

    /// Test if each lane is less than the corresponding lane in `other`.
    fn lanes_lt(self, other: Self) -> Mask;

    /// Test if each lane is greater than the corresponding lane in `other`.
    fn lanes_gt(self, other: Self) -> Mask;

    /// Test if each lane is less than or equal to the corresponding lane in `other`.
    fn lanes_le(self, other: Self) -> Mask;

    /// Test if each lane is greater than or equal to the corresponding lane in `other`.
    fn lanes_ge(self, other: Self) -> Mask;
}

macro_rules! implement_mask_ext {
    { $($vector:ty => $($mask:ty),*;)* } => {
        $( // vector
            $( // mask
                impl MaskExt<$mask> for $vector {
                    #[inline]
                    fn lanes_eq(self, other: Self) -> $mask {
                        unsafe { crate::intrinsics::simd_eq(self, other) }
                    }

                    #[inline]
                    fn lanes_ne(self, other: Self) -> $mask {
                        unsafe { crate::intrinsics::simd_ne(self, other) }
                    }

                    #[inline]
                    fn lanes_lt(self, other: Self) -> $mask {
                        unsafe { crate::intrinsics::simd_lt(self, other) }
                    }

                    #[inline]
                    fn lanes_gt(self, other: Self) -> $mask {
                        unsafe { crate::intrinsics::simd_gt(self, other) }
                    }

                    #[inline]
                    fn lanes_le(self, other: Self) -> $mask {
                        unsafe { crate::intrinsics::simd_le(self, other) }
                    }

                    #[inline]
                    fn lanes_ge(self, other: Self) -> $mask {
                        unsafe { crate::intrinsics::simd_ge(self, other) }
                    }
                }
            )*
        )*
    }
}

implement_mask_ext! {
    crate::u8x8 => crate::masks::wide::m8x8;
    crate::u8x16 => crate::masks::wide::m8x16;
    crate::u8x32 => crate::masks::wide::m8x32;
    crate::u8x64 => crate::masks::wide::m8x64;
    crate::u16x4 => crate::masks::wide::m16x4;
    crate::u16x8 => crate::masks::wide::m16x8;
    crate::u16x16 => crate::masks::wide::m16x16;
    crate::u16x32 => crate::masks::wide::m16x32;
    crate::u32x2 => crate::masks::wide::m32x2;
    crate::u32x4 => crate::masks::wide::m32x4;
    crate::u32x8 => crate::masks::wide::m32x8;
    crate::u32x16 => crate::masks::wide::m32x16;
    crate::u64x2 => crate::masks::wide::m64x2;
    crate::u64x4 => crate::masks::wide::m64x4;
    crate::u64x8 => crate::masks::wide::m64x8;
    crate::u128x2 => crate::masks::wide::m128x2;
    crate::u128x4 => crate::masks::wide::m128x4;
    crate::usizex2 => crate::masks::wide::msizex2;
    crate::usizex4 => crate::masks::wide::msizex4;
    crate::usizex8 => crate::masks::wide::msizex8;

    crate::i8x8 => crate::masks::wide::m8x8;
    crate::i8x16 => crate::masks::wide::m8x16;
    crate::i8x32 => crate::masks::wide::m8x32;
    crate::i8x64 => crate::masks::wide::m8x64;
    crate::i16x4 => crate::masks::wide::m16x4;
    crate::i16x8 => crate::masks::wide::m16x8;
    crate::i16x16 => crate::masks::wide::m16x16;
    crate::i16x32 => crate::masks::wide::m16x32;
    crate::i32x2 => crate::masks::wide::m32x2;
    crate::i32x4 => crate::masks::wide::m32x4;
    crate::i32x8 => crate::masks::wide::m32x8;
    crate::i32x16 => crate::masks::wide::m32x16;
    crate::i64x2 => crate::masks::wide::m64x2;
    crate::i64x4 => crate::masks::wide::m64x4;
    crate::i64x8 => crate::masks::wide::m64x8;
    crate::i128x2 => crate::masks::wide::m128x2;
    crate::i128x4 => crate::masks::wide::m128x4;
    crate::isizex2 => crate::masks::wide::msizex2;
    crate::isizex4 => crate::masks::wide::msizex4;
    crate::isizex8 => crate::masks::wide::msizex8;

    crate::f32x2 => crate::masks::wide::m32x2;
    crate::f32x4 => crate::masks::wide::m32x4;
    crate::f32x8 => crate::masks::wide::m32x8;
    crate::f32x16 => crate::masks::wide::m32x16;
    crate::f64x2 => crate::masks::wide::m64x2;
    crate::f64x4 => crate::masks::wide::m64x4;
    crate::f64x8 => crate::masks::wide::m64x8;
}

macro_rules! implement_mask_ops {
    { $($vector:ty => $mask:ty,)* } => {
        $( // vector
            impl $vector {
                /// Test if each lane is equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_eq(self, other: Self) -> $mask {
                    <$mask>::new_from_inner(MaskExt::lanes_eq(self, other))
                }

                /// Test if each lane is not equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_ne(self, other: Self) -> $mask {
                    <$mask>::new_from_inner(MaskExt::lanes_ne(self, other))
                }

                /// Test if each lane is less than the corresponding lane in `other`.
                #[inline]
                pub fn lanes_lt(self, other: Self) -> $mask {
                    <$mask>::new_from_inner(MaskExt::lanes_lt(self, other))
                }

                /// Test if each lane is greater than the corresponding lane in `other`.
                #[inline]
                pub fn lanes_gt(self, other: Self) -> $mask {
                    <$mask>::new_from_inner(MaskExt::lanes_gt(self, other))
                }

                /// Test if each lane is less than or equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_le(self, other: Self) -> $mask {
                    <$mask>::new_from_inner(MaskExt::lanes_le(self, other))
                }

                /// Test if each lane is greater than or equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_ge(self, other: Self) -> $mask {
                    <$mask>::new_from_inner(MaskExt::lanes_ge(self, other))
                }
            }
        )*
    }
}

implement_mask_ops! {
    crate::u8x8 => crate::mask8x8,
    crate::u8x16 => crate::mask8x16,
    crate::u8x32 => crate::mask8x32,
    crate::u8x64 => crate::mask8x64,
    crate::u16x4 => crate::mask16x4,
    crate::u16x8 => crate::mask16x8,
    crate::u16x16 => crate::mask16x16,
    crate::u16x32 => crate::mask16x32,
    crate::u32x2 => crate::mask32x2,
    crate::u32x4 => crate::mask32x4,
    crate::u32x8 => crate::mask32x8,
    crate::u32x16 => crate::mask32x16,
    crate::u64x2 => crate::mask64x2,
    crate::u64x4 => crate::mask64x4,
    crate::u64x8 => crate::mask64x8,
    crate::u128x2 => crate::mask128x2,
    crate::u128x4 => crate::mask128x4,
    crate::usizex2 => crate::masksizex2,
    crate::usizex4 => crate::masksizex4,
    crate::usizex8 => crate::masksizex8,

    crate::i8x8 => crate::mask8x8,
    crate::i8x16 => crate::mask8x16,
    crate::i8x32 => crate::mask8x32,
    crate::i8x64 => crate::mask8x64,
    crate::i16x4 => crate::mask16x4,
    crate::i16x8 => crate::mask16x8,
    crate::i16x16 => crate::mask16x16,
    crate::i16x32 => crate::mask16x32,
    crate::i32x2 => crate::mask32x2,
    crate::i32x4 => crate::mask32x4,
    crate::i32x8 => crate::mask32x8,
    crate::i32x16 => crate::mask32x16,
    crate::i64x2 => crate::mask64x2,
    crate::i64x4 => crate::mask64x4,
    crate::i64x8 => crate::mask64x8,
    crate::i128x2 => crate::mask128x2,
    crate::i128x4 => crate::mask128x4,
    crate::isizex2 => crate::masksizex2,
    crate::isizex4 => crate::masksizex4,
    crate::isizex8 => crate::masksizex8,

    crate::f32x2 => crate::mask32x2,
    crate::f32x4 => crate::mask32x4,
    crate::f32x8 => crate::mask32x8,
    crate::f32x16 => crate::mask32x16,
    crate::f64x2 => crate::mask64x2,
    crate::f64x4 => crate::mask64x4,
    crate::f64x8 => crate::mask64x8,
}
