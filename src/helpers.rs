use super::{Scalar, ScalarMaybeUndef, EvalResult};

pub trait FalibleScalarExt {
    /// HACK: this function just extracts all bits if `defined != 0`
    /// Mainly used for args of C-functions and we should totally correctly fetch the size
    /// of their arguments
    fn to_bytes(self) -> EvalResult<'static, u128>;
}

impl FalibleScalarExt for Scalar {
    fn to_bytes(self) -> EvalResult<'static, u128> {
        match self {
            Scalar::Bits { bits, size } => {
                assert_ne!(size, 0);
                Ok(bits)
            },
            Scalar::Ptr(_) => err!(ReadPointerAsBytes),
        }
    }
}

impl FalibleScalarExt for ScalarMaybeUndef {
    fn to_bytes(self) -> EvalResult<'static, u128> {
        self.not_undef()?.to_bytes()
    }
}
