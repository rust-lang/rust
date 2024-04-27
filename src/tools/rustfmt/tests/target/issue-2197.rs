// rustfmt-max_width: 79
// rustfmt-wrap_comments: true

/// ```rust
/// unsafe fn sum_sse2(x: i32x4) -> i32 {
///     let x = vendor::_mm_add_epi32(
///         x,
///         vendor::_mm_srli_si128(x.into(), 8).into(),
///     );
///     let x = vendor::_mm_add_epi32(
///         x,
///         vendor::_mm_srli_si128(x.into(), 4).into(),
///     );
///     vendor::_mm_cvtsi128_si32(x)
/// }
/// ```
fn foo() {}
