types! {
    #![unstable(feature = "stdarch_loongarch", issue = "117427")]

    /// LOONGARCH-specific 256-bit wide vector of 32 packed `i8`.
    pub struct v32i8(32 x pub(crate) i8);

    /// LOONGARCH-specific 256-bit wide vector of 16 packed `i16`.
    pub struct v16i16(16 x pub(crate) i16);

    /// LOONGARCH-specific 256-bit wide vector of 8 packed `i32`.
    pub struct v8i32(8 x pub(crate) i32);

    /// LOONGARCH-specific 256-bit wide vector of 4 packed `i64`.
    pub struct v4i64(4 x pub(crate) i64);

    /// LOONGARCH-specific 256-bit wide vector of 32 packed `u8`.
    pub struct v32u8(32 x pub(crate) u8);

    /// LOONGARCH-specific 256-bit wide vector of 16 packed `u16`.
    pub struct v16u16(16 x pub(crate) u16);

    /// LOONGARCH-specific 256-bit wide vector of 8 packed `u32`.
    pub struct v8u32(8 x pub(crate) u32);

    /// LOONGARCH-specific 256-bit wide vector of 4 packed `u64`.
    pub struct v4u64(4 x pub(crate) u64);

    /// LOONGARCH-specific 128-bit wide vector of 8 packed `f32`.
    pub struct v8f32(8 x pub(crate) f32);

    /// LOONGARCH-specific 256-bit wide vector of 4 packed `f64`.
    pub struct v4f64(4 x pub(crate) f64);
}
