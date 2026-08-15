types! {
    #![unstable(feature = "stdarch_loongarch", issue = "117427")]

    /// 256-bit wide integer vector type, LoongArch-specific
    ///
    /// This type is the same as the `__m256i` type defined in `lasxintrin.h`,
    /// representing a 256-bit SIMD register. Usage of this type typically
    /// occurs in conjunction with the `lasx` target features for LoongArch.
    ///
    /// Internally this type may be viewed as:
    ///
    /// * `i8x32` - thirty two `i8` values packed together
    /// * `i16x16` - sixteen `i16` values packed together
    /// * `i32x8` - eight `i32` values packed together
    /// * `i64x4` - four `i64` values packed together
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
    /// Note that this means that an instance of `m256i` typically just means
    /// a "bag of bits" which is left up to interpretation at the point of use.
    ///
    /// Most intrinsics using `m256i` are prefixed with `lasx_` and the integer
    /// types tend to correspond to suffixes like "b", "h", "w" or "d".
    pub struct m256i(4 x i64);

    /// 256-bit wide set of eight `f32` values, LoongArch-specific
    ///
    /// This type is the same as the `__m256` type defined in `lasxintrin.h`,
    /// representing a 256-bit SIMD register which internally consists of
    /// eight packed `f32` instances. Usage of this type typically occurs in
    /// conjunction with the `lasx` target features for LoongArch.
    ///
    /// Note that unlike `m256i`, the integer version of the 256-bit registers,
    /// this `m256` type has *one* interpretation. Each instance of `m256`
    /// always corresponds to `f32x8`, or eight `f32` values packed together.
    ///
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding  between two consecutive elements); however, the
    /// alignment is different and equal to the size of the type. Note that the
    /// ABI for function calls may *not* be the same.
    ///
    /// Most intrinsics using `m256` are prefixed with `lasx_` and are
    /// suffixed with "s".
    pub struct m256(8 x f32);

    /// 256-bit wide set of four `f64` values, LoongArch-specific
    ///
    /// This type is the same as the `__m256d` type defined in `lasxintrin.h`,
    /// representing a 256-bit SIMD register which internally consists of
    /// four packed `f64` instances. Usage of this type typically occurs in
    /// conjunction with the `lasx` target features for LoongArch.
    ///
    /// Note that unlike `m256i`, the integer version of the 256-bit registers,
    /// this `m256d` type has *one* interpretation. Each instance of `m256d`
    /// always corresponds to `f64x4`, or four `f64` values packed together.
    ///
    /// The in-memory representation of this type is the same as the one of an
    /// equivalent array (i.e. the in-memory order of elements is the same, and
    /// there is no padding); however, the alignment is different and equal to
    /// the size of the type. Note that the ABI for function calls may *not* be
    /// the same.
    ///
    /// Most intrinsics using `m256d` are prefixed with `lasx_` and are suffixed
    /// with "d". Not to be confused with "d" which is used for `m256i`.
    pub struct m256d(4 x f64);

}

#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v32i8([i8; 32]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v16i16([i16; 16]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v8i32([i32; 8]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v4i64([i64; 4]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v32u8([u8; 32]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v16u16([u16; 16]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v8u32([u32; 8]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v4u64([u64; 4]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v8f32([f32; 8]);
#[allow(non_camel_case_types)]
#[repr(simd)]
pub(crate) struct __v4f64([f64; 4]);

// These type aliases are provided solely for transitional compatibility.
// They are temporary and will be removed when appropriate.
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v32i8 = m256i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v16i16 = m256i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v8i32 = m256i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v4i64 = m256i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v32u8 = m256i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v16u16 = m256i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v8u32 = m256i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v4u64 = m256i;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v8f32 = m256;
#[allow(non_camel_case_types)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub type v4f64 = m256d;
