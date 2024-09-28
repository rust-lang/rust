/// Get a single value for an argument values array in a determistic way.
/// * `bits`: The number of bits for the type, only 8, 16, 32, 64 are valid values
/// * `index`: The position in the array we are generating for
pub fn value_for_array(bits: u32, index: u32) -> u64 {
    let index = index as usize;
    match bits {
        8 => VALUES_8[index % VALUES_8.len()].into(),
        16 => VALUES_16[index % VALUES_16.len()].into(),
        32 => VALUES_32[index % VALUES_32.len()].into(),
        64 => VALUES_64[index % VALUES_64.len()],
        _ => unimplemented!("value_for_array(bits: {bits}, ..)"),
    }
}

pub const VALUES_8: &[u8] = &[
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0xf0, 0x80, 0x3b, 0xff,
];

pub const VALUES_16: &[u16] = &[
    0x0000, // 0.0
    0x0400, // The smallest normal value.
    0x37ff, // The value just below 0.5.
    0x3800, // 0.5
    0x3801, // The value just above 0.5.
    0x3bff, // The value just below 1.0.
    0x3c00, // 1.0
    0x3c01, // The value just above 1.0.
    0x3e00, // 1.5
    0x4900, // 10
    0x7bff, // The largest finite value.
    0x7c00, // Infinity.
    // NaNs.
    //  - Quiet NaNs
    0x7f23, 0x7e00, //  - Signalling NaNs
    0x7d23, 0x7c01, // Subnormals.
    //  - A recognisable bit pattern.
    0x0012, //  - The largest subnormal value.
    0x03ff, //  - The smallest subnormal value.
    0x0001, // The same values again, but negated.
    0x8000, 0x8400, 0xb7ff, 0xb800, 0xb801, 0xbbff, 0xbc00, 0xbc01, 0xbe00, 0xc900, 0xfbff, 0xfc00,
    0xff23, 0xfe00, 0xfd23, 0xfc01, 0x8012, 0x83ff, 0x8001,
];

pub const VALUES_32: &[u32] = &[
    // Simple values.
    0x00000000, // 0.0
    0x00800000, // The smallest normal value.
    0x3effffff, // The value just below 0.5.
    0x3f000000, // 0.5
    0x3f000001, // The value just above 0.5.
    0x3f7fffff, // The value just below 1.0.
    0x3f800000, // 1.0
    0x3f800001, // The value just above 1.0.
    0x3fc00000, // 1.5
    0x41200000, // 10
    0x7f8fffff, // The largest finite value.
    0x7f800000, // Infinity.
    // NaNs.
    //  - Quiet NaNs
    0x7fd23456, 0x7fc00000, //  - Signalling NaNs
    0x7f923456, 0x7f800001, // Subnormals.
    //  - A recognisable bit pattern.
    0x00123456, //  - The largest subnormal value.
    0x007fffff, //  - The smallest subnormal value.
    0x00000001, // The same values again, but negated.
    0x80000000, 0x80800000, 0xbeffffff, 0xbf000000, 0xbf000001, 0xbf7fffff, 0xbf800000, 0xbf800001,
    0xbfc00000, 0xc1200000, 0xff8fffff, 0xff800000, 0xffd23456, 0xffc00000, 0xff923456, 0xff800001,
    0x80123456, 0x807fffff, 0x80000001,
];

pub const VALUES_64: &[u64] = &[
    // Simple values.
    0x0000000000000000, // 0.0
    0x0010000000000000, // The smallest normal value.
    0x3fdfffffffffffff, // The value just below 0.5.
    0x3fe0000000000000, // 0.5
    0x3fe0000000000001, // The value just above 0.5.
    0x3fefffffffffffff, // The value just below 1.0.
    0x3ff0000000000000, // 1.0
    0x3ff0000000000001, // The value just above 1.0.
    0x3ff8000000000000, // 1.5
    0x4024000000000000, // 10
    0x7fefffffffffffff, // The largest finite value.
    0x7ff0000000000000, // Infinity.
    // NaNs.
    //  - Quiet NaNs
    0x7ff923456789abcd,
    0x7ff8000000000000,
    //  - Signalling NaNs
    0x7ff123456789abcd,
    0x7ff0000000000000,
    // Subnormals.
    //  - A recognisable bit pattern.
    0x000123456789abcd,
    //  - The largest subnormal value.
    0x000fffffffffffff,
    //  - The smallest subnormal value.
    0x0000000000000001,
    // The same values again, but negated.
    0x8000000000000000,
    0x8010000000000000,
    0xbfdfffffffffffff,
    0xbfe0000000000000,
    0xbfe0000000000001,
    0xbfefffffffffffff,
    0xbff0000000000000,
    0xbff0000000000001,
    0xbff8000000000000,
    0xc024000000000000,
    0xffefffffffffffff,
    0xfff0000000000000,
    0xfff923456789abcd,
    0xfff8000000000000,
    0xfff123456789abcd,
    0xfff0000000000000,
    0x800123456789abcd,
    0x800fffffffffffff,
    0x8000000000000001,
];
