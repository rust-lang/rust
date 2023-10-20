types! {
    /// LOONGARCH-specific 128-bit wide vector of 16 packed `i8`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v16i8(pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8);

    /// LOONGARCH-specific 128-bit wide vector of 8 packed `i16`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v8i16(pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16);

    /// LOONGARCH-specific 128-bit wide vector of 4 packed `i32`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v4i32(pub(crate) i32, pub(crate) i32, pub(crate) i32, pub(crate) i32);

    /// LOONGARCH-specific 128-bit wide vector of 2 packed `i64`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v2i64(pub(crate) i64, pub(crate) i64);

    /// LOONGARCH-specific 128-bit wide vector of 16 packed `u8`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v16u8(pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8);

    /// LOONGARCH-specific 128-bit wide vector of 8 packed `u16`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v8u16(pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16);

    /// LOONGARCH-specific 128-bit wide vector of 4 packed `u32`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v4u32(pub(crate) u32, pub(crate) u32, pub(crate) u32, pub(crate) u32);

    /// LOONGARCH-specific 128-bit wide vector of 2 packed `u64`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v2u64(pub(crate) u64, pub(crate) u64);

    /// LOONGARCH-specific 128-bit wide vector of 4 packed `f32`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v4f32(pub(crate) f32, pub(crate) f32, pub(crate) f32, pub(crate) f32);

    /// LOONGARCH-specific 128-bit wide vector of 2 packed `f64`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v2f64(pub(crate) f64, pub(crate) f64);
}
