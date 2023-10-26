types! {
    /// LOONGARCH-specific 256-bit wide vector of 32 packed `i8`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v32i8(
        pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8,
        pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8,
        pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8,
        pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8, pub(crate) i8,
    );

    /// LOONGARCH-specific 256-bit wide vector of 16 packed `i16`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v16i16(
        pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16,
        pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16, pub(crate) i16,
    );

    /// LOONGARCH-specific 256-bit wide vector of 8 packed `i32`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v8i32(pub(crate) i32, pub(crate) i32, pub(crate) i32, pub(crate) i32, pub(crate) i32, pub(crate) i32, pub(crate) i32, pub(crate) i32);

    /// LOONGARCH-specific 256-bit wide vector of 4 packed `i64`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v4i64(pub(crate) i64, pub(crate) i64, pub(crate) i64, pub(crate) i64);

    /// LOONGARCH-specific 256-bit wide vector of 32 packed `u8`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v32u8(
        pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8,
        pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8,
        pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8,
        pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8, pub(crate) u8,
    );

    /// LOONGARCH-specific 256-bit wide vector of 16 packed `u16`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v16u16(
        pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16,
        pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16, pub(crate) u16,
    );

    /// LOONGARCH-specific 256-bit wide vector of 8 packed `u32`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v8u32(pub(crate) u32, pub(crate) u32, pub(crate) u32, pub(crate) u32, pub(crate) u32, pub(crate) u32, pub(crate) u32, pub(crate) u32);

    /// LOONGARCH-specific 256-bit wide vector of 4 packed `u64`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v4u64(pub(crate) u64, pub(crate) u64, pub(crate) u64, pub(crate) u64);

    /// LOONGARCH-specific 128-bit wide vector of 8 packed `f32`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v8f32(pub(crate) f32, pub(crate) f32, pub(crate) f32, pub(crate) f32, pub(crate) f32, pub(crate) f32, pub(crate) f32, pub(crate) f32);

    /// LOONGARCH-specific 256-bit wide vector of 4 packed `f64`.
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub struct v4f64(pub(crate) f64, pub(crate) f64, pub(crate) f64, pub(crate) f64);
}
