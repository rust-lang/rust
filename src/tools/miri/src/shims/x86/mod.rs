use rustc_middle::mir;
use rustc_span::Symbol;
use rustc_target::abi::Size;
use rustc_target::spec::abi::Abi;

use crate::*;
use helpers::bool_to_simd_element;
use shims::foreign_items::EmulateForeignItemResult;

mod aesni;
mod sse;
mod sse2;
mod sse3;
mod sse41;
mod ssse3;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub(super) trait EvalContextExt<'mir, 'tcx: 'mir>:
    crate::MiriInterpCxExt<'mir, 'tcx>
{
    fn emulate_x86_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, EmulateForeignItemResult> {
        let this = self.eval_context_mut();
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.").unwrap();
        match unprefixed_name {
            // Used to implement the `_addcarry_u32` and `_addcarry_u64` functions.
            // Computes a + b with input and output carry. The input carry is an 8-bit
            // value, which is interpreted as 1 if it is non-zero. The output carry is
            // an 8-bit value that will be 0 or 1.
            // https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/addcarry-u32-addcarry-u64.html
            "addcarry.32" | "addcarry.64" => {
                if unprefixed_name == "addcarry.64" && this.tcx.sess.target.arch != "x86_64" {
                    return Ok(EmulateForeignItemResult::NotSupported);
                }

                let [c_in, a, b] = this.check_shim(abi, Abi::Unadjusted, link_name, args)?;
                let c_in = this.read_scalar(c_in)?.to_u8()? != 0;
                let a = this.read_immediate(a)?;
                let b = this.read_immediate(b)?;

                let (sum, overflow1) = this.overflowing_binary_op(mir::BinOp::Add, &a, &b)?;
                let (sum, overflow2) = this.overflowing_binary_op(
                    mir::BinOp::Add,
                    &sum,
                    &ImmTy::from_uint(c_in, a.layout),
                )?;
                let c_out = overflow1 | overflow2;

                this.write_scalar(Scalar::from_u8(c_out.into()), &this.project_field(dest, 0)?)?;
                this.write_immediate(*sum, &this.project_field(dest, 1)?)?;
            }
            // Used to implement the `_subborrow_u32` and `_subborrow_u64` functions.
            // Computes a - b with input and output borrow. The input borrow is an 8-bit
            // value, which is interpreted as 1 if it is non-zero. The output borrow is
            // an 8-bit value that will be 0 or 1.
            // https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/subborrow-u32-subborrow-u64.html
            "subborrow.32" | "subborrow.64" => {
                if unprefixed_name == "subborrow.64" && this.tcx.sess.target.arch != "x86_64" {
                    return Ok(EmulateForeignItemResult::NotSupported);
                }

                let [b_in, a, b] = this.check_shim(abi, Abi::Unadjusted, link_name, args)?;
                let b_in = this.read_scalar(b_in)?.to_u8()? != 0;
                let a = this.read_immediate(a)?;
                let b = this.read_immediate(b)?;

                let (sub, overflow1) = this.overflowing_binary_op(mir::BinOp::Sub, &a, &b)?;
                let (sub, overflow2) = this.overflowing_binary_op(
                    mir::BinOp::Sub,
                    &sub,
                    &ImmTy::from_uint(b_in, a.layout),
                )?;
                let b_out = overflow1 | overflow2;

                this.write_scalar(Scalar::from_u8(b_out.into()), &this.project_field(dest, 0)?)?;
                this.write_immediate(*sub, &this.project_field(dest, 1)?)?;
            }

            name if name.starts_with("sse.") => {
                return sse::EvalContextExt::emulate_x86_sse_intrinsic(
                    this, link_name, abi, args, dest,
                );
            }
            name if name.starts_with("sse2.") => {
                return sse2::EvalContextExt::emulate_x86_sse2_intrinsic(
                    this, link_name, abi, args, dest,
                );
            }
            name if name.starts_with("sse3.") => {
                return sse3::EvalContextExt::emulate_x86_sse3_intrinsic(
                    this, link_name, abi, args, dest,
                );
            }
            name if name.starts_with("ssse3.") => {
                return ssse3::EvalContextExt::emulate_x86_ssse3_intrinsic(
                    this, link_name, abi, args, dest,
                );
            }
            name if name.starts_with("sse41.") => {
                return sse41::EvalContextExt::emulate_x86_sse41_intrinsic(
                    this, link_name, abi, args, dest,
                );
            }
            name if name.starts_with("aesni.") => {
                return aesni::EvalContextExt::emulate_x86_aesni_intrinsic(
                    this, link_name, abi, args, dest,
                );
            }

            _ => return Ok(EmulateForeignItemResult::NotSupported),
        }
        Ok(EmulateForeignItemResult::NeedsJumping)
    }
}

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

#[derive(Copy, Clone)]
enum FloatBinOp {
    /// Arithmetic operation
    Arith(mir::BinOp),
    /// Comparison
    Cmp(FloatCmpOp),
    /// Minimum value (with SSE semantics)
    ///
    /// <https://www.felixcloutier.com/x86/minss>
    /// <https://www.felixcloutier.com/x86/minps>
    /// <https://www.felixcloutier.com/x86/minsd>
    /// <https://www.felixcloutier.com/x86/minpd>
    Min,
    /// Maximum value (with SSE semantics)
    ///
    /// <https://www.felixcloutier.com/x86/maxss>
    /// <https://www.felixcloutier.com/x86/maxps>
    /// <https://www.felixcloutier.com/x86/maxsd>
    /// <https://www.felixcloutier.com/x86/maxpd>
    Max,
}

/// Performs `which` scalar operation on `left` and `right` and returns
/// the result.
fn bin_op_float<'tcx, F: rustc_apfloat::Float>(
    this: &crate::MiriInterpCx<'_, 'tcx>,
    which: FloatBinOp,
    left: &ImmTy<'tcx, Provenance>,
    right: &ImmTy<'tcx, Provenance>,
) -> InterpResult<'tcx, Scalar<Provenance>> {
    match which {
        FloatBinOp::Arith(which) => {
            let res = this.wrapping_binary_op(which, left, right)?;
            Ok(res.to_scalar())
        }
        FloatBinOp::Cmp(which) => {
            let left = left.to_scalar().to_float::<F>()?;
            let right = right.to_scalar().to_float::<F>()?;
            // FIXME: Make sure that these operations match the semantics
            // of cmpps/cmpss/cmppd/cmpsd
            let res = match which {
                FloatCmpOp::Eq => left == right,
                FloatCmpOp::Lt => left < right,
                FloatCmpOp::Le => left <= right,
                FloatCmpOp::Unord => left.is_nan() || right.is_nan(),
                FloatCmpOp::Neq => left != right,
                FloatCmpOp::Nlt => !(left < right),
                FloatCmpOp::Nle => !(left <= right),
                FloatCmpOp::Ord => !left.is_nan() && !right.is_nan(),
            };
            Ok(bool_to_simd_element(res, Size::from_bits(F::BITS)))
        }
        FloatBinOp::Min => {
            let left_scalar = left.to_scalar();
            let left = left_scalar.to_float::<F>()?;
            let right_scalar = right.to_scalar();
            let right = right_scalar.to_float::<F>()?;
            // SSE semantics to handle zero and NaN. Note that `x == F::ZERO`
            // is true when `x` is either +0 or -0.
            if (left == F::ZERO && right == F::ZERO)
                || left.is_nan()
                || right.is_nan()
                || left >= right
            {
                Ok(right_scalar)
            } else {
                Ok(left_scalar)
            }
        }
        FloatBinOp::Max => {
            let left_scalar = left.to_scalar();
            let left = left_scalar.to_float::<F>()?;
            let right_scalar = right.to_scalar();
            let right = right_scalar.to_float::<F>()?;
            // SSE semantics to handle zero and NaN. Note that `x == F::ZERO`
            // is true when `x` is either +0 or -0.
            if (left == F::ZERO && right == F::ZERO)
                || left.is_nan()
                || right.is_nan()
                || left <= right
            {
                Ok(right_scalar)
            } else {
                Ok(left_scalar)
            }
        }
    }
}

/// Performs `which` operation on the first component of `left` and `right`
/// and copies the other components from `left`. The result is stored in `dest`.
fn bin_op_simd_float_first<'tcx, F: rustc_apfloat::Float>(
    this: &mut crate::MiriInterpCx<'_, 'tcx>,
    which: FloatBinOp,
    left: &OpTy<'tcx, Provenance>,
    right: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (left, left_len) = this.operand_to_simd(left)?;
    let (right, right_len) = this.operand_to_simd(right)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, left_len);
    assert_eq!(dest_len, right_len);

    let res0 = bin_op_float::<F>(
        this,
        which,
        &this.read_immediate(&this.project_index(&left, 0)?)?,
        &this.read_immediate(&this.project_index(&right, 0)?)?,
    )?;
    this.write_scalar(res0, &this.project_index(&dest, 0)?)?;

    for i in 1..dest_len {
        this.copy_op(
            &this.project_index(&left, i)?,
            &this.project_index(&dest, i)?,
            /*allow_transmute*/ false,
        )?;
    }

    Ok(())
}

/// Performs `which` operation on each component of `left` and
/// `right`, storing the result is stored in `dest`.
fn bin_op_simd_float_all<'tcx, F: rustc_apfloat::Float>(
    this: &mut crate::MiriInterpCx<'_, 'tcx>,
    which: FloatBinOp,
    left: &OpTy<'tcx, Provenance>,
    right: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (left, left_len) = this.operand_to_simd(left)?;
    let (right, right_len) = this.operand_to_simd(right)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, left_len);
    assert_eq!(dest_len, right_len);

    for i in 0..dest_len {
        let left = this.read_immediate(&this.project_index(&left, i)?)?;
        let right = this.read_immediate(&this.project_index(&right, i)?)?;
        let dest = this.project_index(&dest, i)?;

        let res = bin_op_float::<F>(this, which, &left, &right)?;
        this.write_scalar(res, &dest)?;
    }

    Ok(())
}

/// Horizontaly performs `which` operation on adjacent values of
/// `left` and `right` SIMD vectors and stores the result in `dest`.
fn horizontal_bin_op<'tcx>(
    this: &mut crate::MiriInterpCx<'_, 'tcx>,
    which: mir::BinOp,
    saturating: bool,
    left: &OpTy<'tcx, Provenance>,
    right: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (left, left_len) = this.operand_to_simd(left)?;
    let (right, right_len) = this.operand_to_simd(right)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, left_len);
    assert_eq!(dest_len, right_len);
    assert_eq!(dest_len % 2, 0);

    let middle = dest_len / 2;
    for i in 0..dest_len {
        // `i` is the index in `dest`
        // `j` is the index of the 2-item chunk in `src`
        let (j, src) =
            if i < middle { (i, &left) } else { (i.checked_sub(middle).unwrap(), &right) };
        // `base_i` is the index of the first item of the 2-item chunk in `src`
        let base_i = j.checked_mul(2).unwrap();
        let lhs = this.read_immediate(&this.project_index(src, base_i)?)?;
        let rhs = this.read_immediate(&this.project_index(src, base_i.checked_add(1).unwrap())?)?;

        let res = if saturating {
            Immediate::from(this.saturating_arith(which, &lhs, &rhs)?)
        } else {
            *this.wrapping_binary_op(which, &lhs, &rhs)?
        };

        this.write_immediate(res, &this.project_index(&dest, i)?)?;
    }

    Ok(())
}
