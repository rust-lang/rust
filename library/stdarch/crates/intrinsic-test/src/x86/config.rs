pub fn build_notices(line_prefix: &str) -> String {
    format!(
        "\
{line_prefix}This is a transient test file, not intended for distribution. Some aspects of the
{line_prefix}test are derived from an XML specification, published under the same license as the
{line_prefix}`intrinsic-test` crate.\n
"
    )
}

// Format f16 values (and vectors containing them) in a way that is consistent with C.
pub const F16_FORMATTING_DEF: &str = r#"
#[repr(transparent)]
struct Hex<T>(T);
 "#;

pub const LANE_FUNCTION_HELPERS: &str = r#"
int mm512_extract(__m512i m, int vec_len, int bit_len, int index) {
    int lane_len = 128;
    int max_major_index = vec_len / lane_len;
    int max_minor_index = lane_len / bit_len;

    int major_index = index / max_major_index;
    int minor_index = index % max_minor_index;

    __m128i lane = _mm512_extracti64x2_epi64(m, major_index);

    switch(bit_len){
        case 8:
            return _mm_extract_epi8(lane, minor_index);
        case 16:
            return _mm_extract_epi16(lane, minor_index);
        case 32:
            return _mm_extract_epi32(lane, minor_index);
        case 64:
            return _mm_extract_epi64(lane, minor_index);
    }
}

int _mm512_extract_intrinsic_test_epi8(__m512i m, int lane) {
    return mm512_extract(m, 512, 8, lane)
}

int _mm512_extract_intrinsic_test_epi16(__m512i m, int lane) {
    return mm512_extract(m, 512, 16, lane)
}

int mm512_extract_intrinsic_test_epi16(__m512i m, int lane) {
    return mm512_extract(m, 512, 16, lane)
}

int mm512_extract_intrinsic_test_epi64(__m512i m, int lane) {
    return mm512_extract(m, 512, 64, lane)
}

int mm64_extract_intrinsic_test_epi8(__m64 m, int lane) {
    int real_lane_shift = lane / 2;
    int real_bit_shift = (lane % 2) * 8;
    int result = _mm_extract_pi16(m, lane / 2);
    return (result >> real_bit_shift);
}

int mm64_extract_intrinsic_test_epi32(__m64 m, int lane) {
    int bit_shift_amount = lane * 32;
    return _m_to_int(m >> bit_shift_amount);
}
"#;

pub const X86_CONFIGURATIONS: &str = r#"
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_avx512_bf16))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_avx512_f16))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_rtm))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_rtm))]
#![cfg_attr(target_arch = "x86_64", feature(x86_amx_intrinsics))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512_f16))]
#![feature(fmt_helpers_for_derive)]
"#;
