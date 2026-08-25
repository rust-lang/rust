// rustfmt-wrap_comments: true

/// Although the indentation of the skipped method is off, it shouldn't be
/// changed.
///
/// ```
/// pub unsafe fn _mm256_shufflehi_epi16(a: __m256i, imm8: i32) -> __m256i {
///     let imm8 = (imm8 & 0xFF) as u8;
///     let a = a.as_i16x16();
///     macro_rules! shuffle_done {
///         ($x01:expr, $x23:expr, $x45:expr, $x67:expr) => {
///             #[cfg_attr(rustfmt, rustfmt_skip)]
///       simd_shuffle16(a, a, [
///           0, 1, 2, 3, 4+$x01, 4+$x23, 4+$x45, 4+$x67,
///           8, 9, 10, 11, 12+$x01, 12+$x23, 12+$x45, 12+$x67
///       ]);
///         };
///     }
/// }
/// ```
pub unsafe fn _mm256_shufflehi_epi16(a: __m256i, imm8: i32) -> __m256i {
    let imm8 = (imm8 & 0xFF) as u8;
    let a = a.as_i16x16();
    macro_rules! shuffle_done {
        ($x01:expr, $x23:expr, $x45:expr, $x67:expr) => {
            #[cfg_attr(rustfmt, rustfmt_skip)]
                         simd_shuffle16(a, a, [
                             0, 1, 2, 3, 4+$x01, 4+$x23, 4+$x45, 4+$x67,
                             8, 9, 10, 11, 12+$x01, 12+$x23, 12+$x45, 12+$x67
                         ]);
        };
    }
}

/// The skipped method shouldn't right-shift
pub unsafe fn _mm256_shufflehi_epi32(a: __m256i, imm8: i32) -> __m256i {
    let imm8 = (imm8 & 0xFF) as u8;
    let a = a.as_i16x16();
    macro_rules! shuffle_done {
        ($x01:expr, $x23:expr, $x45:expr, $x67:expr) => {
            #[cfg_attr(rustfmt, rustfmt_skip)]
            simd_shuffle32(a, a, [
                0, 1, 2, 3, 4+$x01, 4+$x23, 4+$x45, 4+$x67,
                8, 9, 10, 11, 12+$x01, 12+$x23, 12+$x45, 12+$x67
            ]);
        };
    }
}
