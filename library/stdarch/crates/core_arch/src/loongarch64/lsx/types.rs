types! {
    #![unstable(feature = "stdarch_loongarch", issue = "117427")]

    /// 128-bit wide integer vector type, LoongArch-specific
    ///
    /// This type is the same as the `__m128i` type defined in `lsxintrin.h`,
    /// representing a 128-bit SIMD register. Usage of this type typically
    /// occurs in conjunction with the `lsx` and higher target features for
    /// LoongArch.
    ///
    /// Internally this type may be viewed as:
    ///
    /// * `i8x16` - sixteen `i8` values packed together
    /// * `i16x8` - eight `i16` values packed together
    /// * `i32x4` - four `i32` values packed together
    /// * `i64x2` - two `i64` values packed together
    ///
    /// (as well as unsigned versions). Each intrinsic may interpret the
    /// internal bits differently, check the documentation of the intrinsic
    /// to see how it's being used.
    ///
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
    ///
    /// Note that this means that an instance of `m128i` typically just means
    /// a "bag of bits" which is left up to interpretation at the point of use.
    ///
    /// Most intrinsics using `m128i` are prefixed with `lsx_` and the integer
    /// types tend to correspond to suffixes like "b", "h", "w" or "d".
    pub struct m128i(2 x i64);

    /// 128-bit wide set of four `f32` values, LoongArch-specific
    ///
    /// This type is the same as the `__m128` type defined in `lsxintrin.h`,
    /// representing a 128-bit SIMD register which internally consists of
    /// four packed `f32` instances. Usage of this type typically occurs in
    /// conjunction with the `lsx` and higher target features for LoongArch.
    ///
    /// Note that unlike `m128i`, the integer version of the 128-bit registers,
    /// this `m128` type has *one* interpretation. Each instance of `m128`
    /// corresponds to `f32x4`, or four `f32` values packed together.
    ///
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
    ///
    /// Most intrinsics using `m128` are prefixed with `lsx_` and are suffixed
    /// with "s".
    pub struct m128(4 x f32);

    /// 128-bit wide set of two `f64` values, LoongArch-specific
    ///
    /// This type is the same as the `__m128d` type defined in `lsxintrin.h`,
    /// representing a 128-bit SIMD register which internally consists of
    /// two packed `f64` instances. Usage of this type typically occurs in
    /// conjunction with the `lsx` and higher target features for LoongArch.
    ///
    /// Note that unlike `m128i`, the integer version of the 128-bit registers,
    /// this `m128d` type has *one* interpretation. Each instance of `m128d`
    /// always corresponds to `f64x2`, or two `f64` values packed together.
    ///
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
    ///
    /// Most intrinsics using `m128d` are prefixed with `lsx_` and are suffixed
    /// with "d". Not to be confused with "d" which is used for `m128i`.
    pub struct m128d(2 x f64);
}

#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v16i8([i8; 16]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v8i16([i16; 8]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v4i32([i32; 4]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v2i64([i64; 2]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v16u8([u8; 16]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v8u16([u16; 8]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v4u32([u32; 4]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v2u64([u64; 2]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v4f32([f32; 4]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v2f64([f64; 2]);

// These type aliases are provided solely for transitional compatibility.
// They are temporary and will be removed when appropriate.
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v16i8 = m128i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v8i16 = m128i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v4i32 = m128i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v2i64 = m128i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v16u8 = m128i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v8u16 = m128i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v4u32 = m128i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v2u64 = m128i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v4f32 = m128;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v2f64 = m128d;
