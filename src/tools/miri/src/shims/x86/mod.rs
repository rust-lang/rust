use crate::InterpResult;

pub(super) mod sse;
pub(super) mod sse2;

/// Floating point comparison operation
///
/// <https://www.felixcloutier.com/x86/cmpss>
/// <https://www.felixcloutier.com/x86/cmpps>
/// <https://www.felixcloutier.com/x86/cmpsd>
/// <https://www.felixcloutier.com/x86/cmppd>
#[derive(Copy, Clone)]
enum FloatCmpOp {
    Eq,
    Lt,
    Le,
    Unord,
    Neq,
    /// Not less-than
    Nlt,
    /// Not less-or-equal
    Nle,
    /// Ordered, i.e. neither of them is NaN
    Ord,
}

impl FloatCmpOp {
    /// Convert from the `imm` argument used to specify the comparison
    /// operation in intrinsics such as `llvm.x86.sse.cmp.ss`.
    fn from_intrinsic_imm(imm: i8, intrinsic: &str) -> InterpResult<'_, Self> {
        match imm {
            0 => Ok(Self::Eq),
            1 => Ok(Self::Lt),
            2 => Ok(Self::Le),
            3 => Ok(Self::Unord),
            4 => Ok(Self::Neq),
            5 => Ok(Self::Nlt),
            6 => Ok(Self::Nle),
            7 => Ok(Self::Ord),
            imm => {
                throw_unsup_format!("invalid `imm` parameter of {intrinsic}: {imm}");
            }
        }
    }
}
