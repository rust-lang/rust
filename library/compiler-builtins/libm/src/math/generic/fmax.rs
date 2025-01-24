use super::super::Float;

#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmax<F: Float>(x: F, y: F) -> F {
    // IEEE754 says: maxNum(x, y) is the canonicalized number y if x < y, x if y < x, the
    // canonicalized number if one operand is a number and the other a quiet NaN. Otherwise it
    // is either x or y, canonicalized (this means results might differ among implementations).
    // When either x or y is a signalingNaN, then the result is according to 6.2.
    //
    // Since we do not support sNaN in Rust yet, we do not need to handle them.
    // FIXME(nagisa): due to https://bugs.llvm.org/show_bug.cgi?id=33303 we canonicalize by
    // multiplying by 1.0. Should switch to the `canonicalize` when it works.
    (if x.is_nan() || x < y { y } else { x }) * F::ONE
}
