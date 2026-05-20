use itertools::Itertools as _;

use crate::common::{
    PASSES,
    intrinsic_helpers::{IntrinsicType, IntrinsicTypeDefinition, Sign, SimdLen, TypeKind},
};

/// Maximum size of a SVE vector
pub const MAX_SVE_BITS: u32 = 2048;

/// Returns a string with the name of the static variable containing test values for intrinsic
/// arguments of this type.
pub fn test_values_array_name<T: IntrinsicTypeDefinition>(ty: &T) -> String {
    format!(
        "{ty}_{load_size}",
        ty = ty.rust_scalar_type().to_uppercase(),
        load_size = test_values_array_length(&ty),
    )
}

/// Returns the elements used in the test value arrays in `gen_arg_rust`. Uses the
/// `test_values_array_length` fn to determine the number of values that
/// `ArgumentList::gen_arg_rust` expects and `ArgumentList::load_values_rust` needs.
///
/// Each value in the array starts as a bit pattern from `bit_pattern_for_test_values_array`
/// which is then printed as a hex value in the generated code (and if identified as a negative
/// value, with the appropriate minus and corrected hex pattern). Calls to `fN::from_bits` are
/// generated for floats.
pub fn test_values_array(ty: &IntrinsicType) -> String {
    let (bit_len, kind) = match ty {
        IntrinsicType {
            kind: TypeKind::Float,
            bit_len: Some(bit_len),
            ..
        } => (*bit_len, TypeKind::Float),
        IntrinsicType {
            kind: TypeKind::Vector,
            ..
        } => (32, TypeKind::Vector),
        IntrinsicType {
            kind,
            bit_len: Some(bit_len),
            ..
        } => (*bit_len, *kind),
        _ => unimplemented!(),
    };

    format!(
        "[{}]",
        (0..test_values_array_length(ty)).format_with(",", |i, fmt| {
            let src = bit_pattern_for_test_values_array(bit_len, i);
            assert!(src == 0 || src.ilog2() < bit_len);
            match kind {
                TypeKind::Float => fmt(&format_args!("f{bit_len}::from_bits({src:#x})")),
                TypeKind::Vector | TypeKind::Int(Sign::Signed) if (src >> (bit_len - 1)) != 0 => {
                    // `src` is a two's complement representation of a negative value.
                    let mask = !0u64 >> (64 - bit_len);
                    let ones_compl = src ^ mask;
                    let twos_compl = ones_compl + 1;
                    fmt(&format_args!("-{twos_compl:#x}"))
                }
                _ => fmt(&format_args!("{src:#x}")),
            }
        })
    )
}

/// Returns the number of values that need to be in an array of test values such that there can be
/// `num_loads` distinct windows for a given vector of type `ty`.
///
/// For example, vectors of type `uint32x2x2_t` load four values (`2 x 2`) and so to support
/// `num_loads=10` distinct windows, the total length of the array of test values must be
/// `(2 x 2) + 10 - 1`:
///
/// ```text
/// [0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD]
///  ^^^^^^^^^^^^^^^^^^ first window
///       ^^^^^^^^^^^^^^^^^^ second window
///                                   10th window ^^^^^^^^^^^^^^^^^^
/// ```
///
/// For scalable vectors (only SVE is currently supported), assume that the length of the vector is
/// the maximum supported by the architecture.
pub fn test_values_array_length(ty: &IntrinsicType) -> u32 {
    let IntrinsicType {
        simd_len, vec_len, ..
    } = ty;

    let simd_len = simd_len.map_or(1, |v| {
        if let SimdLen::Fixed(n) = v {
            n
        } else {
            MAX_SVE_BITS / ty.inner_size()
        }
    });
    let vec_len = vec_len.unwrap_or(1);

    (simd_len * vec_len) + PASSES - 1
}

/// Returns a bit pattern for a value being output into a array of test values. Bit patterns come
/// from one of many constant arrays of test values. The specific constant array used depends on
/// the number of bits - `bits` - of the type having test values generated for it. This function
/// is called repeatedly with incrementing values of `index` to produce an entire array of test
/// values.
///
/// Each constant array of bit patterns should ideally be at least the length of the largest array
/// of test values that will be requested (e.g. 51 for a `poly8x8x4` when `PASSES=20`:
/// `(8 * 4) + 20 - 1`), otherwise values will be repeated.
pub fn bit_pattern_for_test_values_array(bits: u32, index: u32) -> u64 {
    let index = index as usize;
    match bits {
        1 => BIT_PATTERNS_8[index % 2].into(),
        2 => BIT_PATTERNS_8[index % 4].into(),
        3 => BIT_PATTERNS_8[index % 8].into(),
        4 => BIT_PATTERNS_8[index % 16].into(),
        5 => BIT_PATTERNS_5[index % BIT_PATTERNS_5.len()].into(),
        6 => BIT_PATTERNS_6[index % BIT_PATTERNS_6.len()].into(),
        7 => BIT_PATTERNS_7[index % BIT_PATTERNS_7.len()].into(),
        8 => BIT_PATTERNS_8[index % BIT_PATTERNS_8.len()].into(),
        16 => BIT_PATTERNS_16[index % BIT_PATTERNS_16.len()].into(),
        32 => BIT_PATTERNS_32[index % BIT_PATTERNS_32.len()].into(),
        64 => BIT_PATTERNS_64[index % BIT_PATTERNS_64.len()],
        _ => unimplemented!("bit_pattern_for_test_values_array(bits: {bits}, ..)"),
    }
}

pub const BIT_PATTERNS_5: &[u8] = &[
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x019, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
    0x1f,
];

pub const BIT_PATTERNS_6: &[u8] = &[
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x039, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e,
    0x3f,
];

pub const BIT_PATTERNS_7: &[u8] = &[
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x079, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e,
    0x7f,
];

pub const BIT_PATTERNS_8: &[u8] = &[
    0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x10, 0x11,
    0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21,
    0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31,
    0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x40, 0x41,
    0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x50, 0x51,
    0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f, 0x60, 0x61,
    0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71,
    0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f, 0x80, 0x81,
    0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90, 0x91,
    0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f, 0xa0, 0xa1,
    0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf, 0xb0, 0xb1,
    0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf, 0xc0, 0xc1,
    0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf, 0xd0, 0xd1,
    0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf, 0xe0, 0xe1,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef, 0xf0, 0xf1,
    0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
];

#[rustfmt::skip]
pub const BIT_PATTERNS_16: &[u16] = &[
    // Simple values:
    // 0.0
    0x0000,
    // The smallest normal value
    0x0400,
    // The value just below 0.5
    0x37ff,
    // 0.5
    0x3800,
    // The value just above 0.5
    0x3801,
    // The value just below 1.0
    0x3bff,
    // 1.0
    0x3c00,
    // The value just above 1.0
    0x3c01,
    // 1.5
    0x3e00,
    // 10
    0x4900,
    // The largest finite value
    0x7bff,
    // Infinity.
    0x7c00,

    // NaNs:
    // Quiet NaNs
    0x7f23,
    0x7e00,
    // Signalling NaNs
    0x7d23,
    0x7c01,

    // Subnormals:
    // A recognisable bit pattern
    0x0012,
    // The largest subnormal value
    0x03ff,
    // The smallest subnormal value
    0x0001,

    // Other values:
    // Above values, negated
    0x8000, 0x8400, 0xb7ff, 0xb800, 0xb801, 0xbbff, 0xbc00, 0xbc01, 0xbe00, 0xc900, 0xfbff, 0xfc00,
    0xff23, 0xfe00, 0xfd23, 0xfc01, 0x8012, 0x83ff, 0x8001,
    // Random values 
    0xfc00, 0xc000, 0x5140, 0x5800, 0x63d2, 0x5630, 0x3560, 0x9191, 0x4178, 0x6212, 0x67d0, 0x3312,
    0x4cef, 0x4973, 0x3ecc, 0x5166, 0x4d80, 0x6248, 0x46fd, 0x39c4, 0x39c5, 0x4866, 0x6050, 0x498e,
    0x4a0f,
    // Previous values in a different order
    0x3555, 0xfc00, 0xc000, 0x9191, 0x5140, 0x5800, 0x8001, 0x83ff, 0x63d2, 0x5630, 0x3560, 0x4178,
    0x7d23, 0x7c01, 0x0012, 0xb800, 0x03ff, 0x0001, 0x7e00, 0x7f23, 0x8000, 0x8400, 0xb7ff, 0xb801,
    0x3312, 0x4cef, 0x4973, 0x39c4, 0x3ecc, 0x5166, 0x67d0, 0x6212, 0x4d80, 0x6248, 0x46fd, 0x39c5,
    0xbc01, 0xbe00, 0xc900, 0xfc01, 0xfbff, 0xfc00, 0xbc00, 0xbbff, 0xff23, 0xfe00, 0xfd23, 0x8012,
    0x37ff, 0x3800, 0x3801, 0x7bff, 0x3bff, 0x3c00, 0x0400, 0x0000, 0x3c01, 0x3e00, 0x4900, 0x7c00,
    0x498e, 0x4a0f, 0x6050, 0x4866,

    // Specific values:
    // As close to 1/3 as possible.
    0x3555,
];

#[rustfmt::skip]
pub const BIT_PATTERNS_32: &[u32] = &[
    // Simple values:
    // 0.0
    0x00000000,
    // The smallest normal value
    0x00800000,
    // The value just below 0.5
    0x3effffff,
    // 0.5
    0x3f000000,
    // The value just above 0.5
    0x3f000001,
    // The value just below 1.0
    0x3f7fffff,
    // 1.0
    0x3f800000,
    // The value just above 1.0
    0x3f800001,
    // 1.5
    0x3fc00000,
    // 10
    0x41200000,
    // The largest finite value
    0x7f8fffff,
    // Infinity
    0x7f800000,

    // NaNs:
    // Quiet NaNs
    0x7fd23456,
    0x7fc00000,
    // Signalling NaNs
    0x7f923456,
    0x7f800001,

    // Subnormals:
    // A recognisable bit pattern
    0x00123456,
    // The largest subnormal value
    0x007fffff,
    // The smallest subnormal value
    0x00000001,

    // Other values:
    // Above values, negated
    0x80000000, 0x80800000, 0xbeffffff, 0xbf000000, 0xbf000001, 0xbf7fffff, 0xbf800000, 0xbf800001,
    0xbfc00000, 0xc1200000, 0xff8fffff, 0xff800000, 0xffd23456, 0xffc00000, 0xff923456, 0xff800001,
    0x80123456, 0x807fffff, 0x80000001, 0x80123456, 0x807fffff, 0x80000001,
    // Random values
    0x4205cccd, 0x4229178D, 0x42C6A0C5, 0x3B3302F7, 0x3F9DF45E, 0x41DAA3D7, 0x47C3501D, 0xC3889333,
    0xC2C675C3, 0xC69C449A, 0xC341FD71, 0xC502DFD7, 0xBBB43958, 0x3EE24DD3, 0x42B1C28F, 0x42F06666,
    0x45D379C3, 0x44637148, 0x3CBBECAB, 0x4113EDFA, 0x444B22F2, 0x1FD93A96, 0x9921055F, 0xFF626925,
    
    // Specific values:
    // Approximately Pi
    0x40490fdb,
    // Approximately 1/3
    0x3eaaaaab,
];

#[rustfmt::skip]
pub const BIT_PATTERNS_64: &[u64] = &[
    // Simple values:
    // 0.0
    0x0000000000000000,
    // The smallest normal value
    0x0010000000000000,
    // The value just below 0.5
    0x3fdfffffffffffff,
    // 0.5
    0x3fe0000000000000,
    // The value just above 0.5
    0x3fe0000000000001,
    // The value just below 1.0
    0x3fefffffffffffff,
    // 1.0
    0x3ff0000000000000,
    // The value just above 1.0
    0x3ff0000000000001,
    // 1.5
    0x3ff8000000000000,
    // 10
    0x4024000000000000,
    // The largest finite value
    0x7fefffffffffffff,
    // Infinity
    0x7ff0000000000000,

    // NaNs:
    // Quiet NaNs
    0x7ff923456789abcd, 0x7ff8000000000000,
    // Signalling NaNs
    0x7ff123456789abcd, 0x7ff0000000000000,

    // Subnormals:
    // A recognisable bit pattern
    0x000123456789abcd,
    // The largest subnormal value
    0x000fffffffffffff,
    // The smallest subnormal value
    0x0000000000000001,

    // Other values:
    // Above values, negated
    0x8000000000000000, 0x8010000000000000, 0xbfdfffffffffffff, 0xbfe0000000000000,
    0xbfe0000000000001, 0xbfefffffffffffff, 0xbff0000000000000, 0xbff0000000000001,
    0xbff8000000000000, 0xc024000000000000, 0xffefffffffffffff, 0xfff0000000000000,
    0xfff923456789abcd, 0xfff8000000000000, 0xfff123456789abcd, 0xfff0000000000000,
    0x800123456789abcd, 0x800fffffffffffff, 0x8000000000000001,

    // Specific values:
    // Pi
    0x400921FB54442D18,
    // Approximately 1/3
    0x3fd5555555555555,
];
