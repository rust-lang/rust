use rand::Rng as _;

use rustc_apfloat::{ieee::Single, Float as _};
use rustc_middle::{mir, ty};
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

#[derive(Copy, Clone)]
enum FloatBinOp {
    /// Arithmetic operation
    Arith(mir::BinOp),
    /// Comparison
    ///
    /// The semantics of this operator is a case distinction: we compare the two operands,
    /// and then we return one of the four booleans `gt`, `lt`, `eq`, `unord` depending on
    /// which class they fall into.
    ///
    /// AVX supports all 16 combinations, SSE only a subset
    ///
    /// <https://www.felixcloutier.com/x86/cmpss>
    /// <https://www.felixcloutier.com/x86/cmpps>
    /// <https://www.felixcloutier.com/x86/cmpsd>
    /// <https://www.felixcloutier.com/x86/cmppd>
    Cmp {
        /// Result when lhs < rhs
        gt: bool,
        /// Result when lhs > rhs
        lt: bool,
        /// Result when lhs == rhs
        eq: bool,
        /// Result when lhs is NaN or rhs is NaN
        unord: bool,
    },
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

impl FloatBinOp {
    /// Convert from the `imm` argument used to specify the comparison
    /// operation in intrinsics such as `llvm.x86.sse.cmp.ss`.
    fn cmp_from_imm<'tcx>(
        this: &crate::MiriInterpCx<'_, 'tcx>,
        imm: i8,
        intrinsic: Symbol,
    ) -> InterpResult<'tcx, Self> {
        // Only bits 0..=4 are used, remaining should be zero.
        if imm & !0b1_1111 != 0 {
            throw_unsup_format!("invalid `imm` parameter of {intrinsic}: 0x{imm:x}");
        }
        // Bit 4 specifies whether the operation is quiet or signaling, which
        // we do not care in Miri.
        // Bits 0..=2 specifies the operation.
        // `gt` indicates the result to be returned when the LHS is strictly
        // greater than the RHS, and so on.
        let (gt, lt, eq, mut unord) = match imm & 0b111 {
            // Equal
            0x0 => (false, false, true, false),
            // Less-than
            0x1 => (false, true, false, false),
            // Less-or-equal
            0x2 => (false, true, true, false),
            // Unordered (either is NaN)
            0x3 => (false, false, false, true),
            // Not equal
            0x4 => (true, true, false, true),
            // Not less-than
            0x5 => (true, false, true, true),
            // Not less-or-equal
            0x6 => (true, false, false, true),
            // Ordered (neither is NaN)
            0x7 => (true, true, true, false),
            _ => unreachable!(),
        };
        // When bit 3 is 1 (only possible in AVX), unord is toggled.
        if imm & 0b1000 != 0 {
            this.expect_target_feature_for_intrinsic(intrinsic, "avx")?;
            unord = !unord;
        }
        Ok(Self::Cmp { gt, lt, eq, unord })
    }
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
        FloatBinOp::Cmp { gt, lt, eq, unord } => {
            let left = left.to_scalar().to_float::<F>()?;
            let right = right.to_scalar().to_float::<F>()?;

            let res = match left.partial_cmp(&right) {
                None => unord,
                Some(std::cmp::Ordering::Less) => lt,
                Some(std::cmp::Ordering::Equal) => eq,
                Some(std::cmp::Ordering::Greater) => gt,
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

#[derive(Copy, Clone)]
enum FloatUnaryOp {
    /// sqrt(x)
    ///
    /// <https://www.felixcloutier.com/x86/sqrtss>
    /// <https://www.felixcloutier.com/x86/sqrtps>
    Sqrt,
    /// Approximation of 1/x
    ///
    /// <https://www.felixcloutier.com/x86/rcpss>
    /// <https://www.felixcloutier.com/x86/rcpps>
    Rcp,
    /// Approximation of 1/sqrt(x)
    ///
    /// <https://www.felixcloutier.com/x86/rsqrtss>
    /// <https://www.felixcloutier.com/x86/rsqrtps>
    Rsqrt,
}

/// Performs `which` scalar operation on `op` and returns the result.
#[allow(clippy::arithmetic_side_effects)] // floating point operations without side effects
fn unary_op_f32<'tcx>(
    this: &mut crate::MiriInterpCx<'_, 'tcx>,
    which: FloatUnaryOp,
    op: &ImmTy<'tcx, Provenance>,
) -> InterpResult<'tcx, Scalar<Provenance>> {
    match which {
        FloatUnaryOp::Sqrt => {
            let op = op.to_scalar();
            // FIXME using host floats
            Ok(Scalar::from_u32(f32::from_bits(op.to_u32()?).sqrt().to_bits()))
        }
        FloatUnaryOp::Rcp => {
            let op = op.to_scalar().to_f32()?;
            let div = (Single::from_u128(1).value / op).value;
            // Apply a relative error with a magnitude on the order of 2^-12 to simulate the
            // inaccuracy of RCP.
            let res = apply_random_float_error(this, div, -12);
            Ok(Scalar::from_f32(res))
        }
        FloatUnaryOp::Rsqrt => {
            let op = op.to_scalar().to_u32()?;
            // FIXME using host floats
            let sqrt = Single::from_bits(f32::from_bits(op).sqrt().to_bits().into());
            let rsqrt = (Single::from_u128(1).value / sqrt).value;
            // Apply a relative error with a magnitude on the order of 2^-12 to simulate the
            // inaccuracy of RSQRT.
            let res = apply_random_float_error(this, rsqrt, -12);
            Ok(Scalar::from_f32(res))
        }
    }
}

/// Disturbes a floating-point result by a relative error on the order of (-2^scale, 2^scale).
#[allow(clippy::arithmetic_side_effects)] // floating point arithmetic cannot panic
fn apply_random_float_error<F: rustc_apfloat::Float>(
    this: &mut crate::MiriInterpCx<'_, '_>,
    val: F,
    err_scale: i32,
) -> F {
    let rng = this.machine.rng.get_mut();
    // generates rand(0, 2^64) * 2^(scale - 64) = rand(0, 1) * 2^scale
    let err =
        F::from_u128(rng.gen::<u64>().into()).value.scalbn(err_scale.checked_sub(64).unwrap());
    // give it a random sign
    let err = if rng.gen::<bool>() { -err } else { err };
    // multiple the value with (1+err)
    (val * (F::from_u128(1).value + err).value).value
}

/// Performs `which` operation on the first component of `op` and copies
/// the other components. The result is stored in `dest`.
fn unary_op_ss<'tcx>(
    this: &mut crate::MiriInterpCx<'_, 'tcx>,
    which: FloatUnaryOp,
    op: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (op, op_len) = this.operand_to_simd(op)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, op_len);

    let res0 = unary_op_f32(this, which, &this.read_immediate(&this.project_index(&op, 0)?)?)?;
    this.write_scalar(res0, &this.project_index(&dest, 0)?)?;

    for i in 1..dest_len {
        this.copy_op(
            &this.project_index(&op, i)?,
            &this.project_index(&dest, i)?,
        )?;
    }

    Ok(())
}

/// Performs `which` operation on each component of `op`, storing the
/// result is stored in `dest`.
fn unary_op_ps<'tcx>(
    this: &mut crate::MiriInterpCx<'_, 'tcx>,
    which: FloatUnaryOp,
    op: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (op, op_len) = this.operand_to_simd(op)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, op_len);

    for i in 0..dest_len {
        let op = this.read_immediate(&this.project_index(&op, i)?)?;
        let dest = this.project_index(&dest, i)?;

        let res = unary_op_f32(this, which, &op)?;
        this.write_scalar(res, &dest)?;
    }

    Ok(())
}

// Rounds the first element of `right` according to `rounding`
// and copies the remaining elements from `left`.
fn round_first<'tcx, F: rustc_apfloat::Float>(
    this: &mut crate::MiriInterpCx<'_, 'tcx>,
    left: &OpTy<'tcx, Provenance>,
    right: &OpTy<'tcx, Provenance>,
    rounding: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (left, left_len) = this.operand_to_simd(left)?;
    let (right, right_len) = this.operand_to_simd(right)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, left_len);
    assert_eq!(dest_len, right_len);

    let rounding = rounding_from_imm(this.read_scalar(rounding)?.to_i32()?)?;

    let op0: F = this.read_scalar(&this.project_index(&right, 0)?)?.to_float()?;
    let res = op0.round_to_integral(rounding).value;
    this.write_scalar(
        Scalar::from_uint(res.to_bits(), Size::from_bits(F::BITS)),
        &this.project_index(&dest, 0)?,
    )?;

    for i in 1..dest_len {
        this.copy_op(
            &this.project_index(&left, i)?,
            &this.project_index(&dest, i)?,
        )?;
    }

    Ok(())
}

// Rounds all elements of `op` according to `rounding`.
fn round_all<'tcx, F: rustc_apfloat::Float>(
    this: &mut crate::MiriInterpCx<'_, 'tcx>,
    op: &OpTy<'tcx, Provenance>,
    rounding: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (op, op_len) = this.operand_to_simd(op)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(dest_len, op_len);

    let rounding = rounding_from_imm(this.read_scalar(rounding)?.to_i32()?)?;

    for i in 0..dest_len {
        let op: F = this.read_scalar(&this.project_index(&op, i)?)?.to_float()?;
        let res = op.round_to_integral(rounding).value;
        this.write_scalar(
            Scalar::from_uint(res.to_bits(), Size::from_bits(F::BITS)),
            &this.project_index(&dest, i)?,
        )?;
    }

    Ok(())
}

/// Gets equivalent `rustc_apfloat::Round` from rounding mode immediate of
/// `round.{ss,sd,ps,pd}` intrinsics.
fn rounding_from_imm<'tcx>(rounding: i32) -> InterpResult<'tcx, rustc_apfloat::Round> {
    // The fourth bit of `rounding` only affects the SSE status
    // register, which cannot be accessed from Miri (or from Rust,
    // for that matter), so we can ignore it.
    match rounding & !0b1000 {
        // When the third bit is 0, the rounding mode is determined by the
        // first two bits.
        0b000 => Ok(rustc_apfloat::Round::NearestTiesToEven),
        0b001 => Ok(rustc_apfloat::Round::TowardNegative),
        0b010 => Ok(rustc_apfloat::Round::TowardPositive),
        0b011 => Ok(rustc_apfloat::Round::TowardZero),
        // When the third bit is 1, the rounding mode is determined by the
        // SSE status register. Since we do not support modifying it from
        // Miri (or Rust), we assume it to be at its default mode (round-to-nearest).
        0b100..=0b111 => Ok(rustc_apfloat::Round::NearestTiesToEven),
        rounding => throw_unsup_format!("unsupported rounding mode 0x{rounding:02x}"),
    }
}

/// Converts each element of `op` from floating point to signed integer.
///
/// When the input value is NaN or out of range, fall back to minimum value.
///
/// If `op` has more elements than `dest`, extra elements are ignored. If `op`
/// has less elements than `dest`, the rest is filled with zeros.
fn convert_float_to_int<'tcx>(
    this: &mut crate::MiriInterpCx<'_, 'tcx>,
    op: &OpTy<'tcx, Provenance>,
    rnd: rustc_apfloat::Round,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (op, op_len) = this.operand_to_simd(op)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    // Output must be *signed* integers.
    assert!(matches!(dest.layout.field(this, 0).ty.kind(), ty::Int(_)));

    for i in 0..op_len.min(dest_len) {
        let op = this.read_immediate(&this.project_index(&op, i)?)?;
        let dest = this.project_index(&dest, i)?;

        let res = this.float_to_int_checked(&op, dest.layout, rnd)?.unwrap_or_else(|| {
            // Fallback to minimum acording to SSE/AVX semantics.
            ImmTy::from_int(dest.layout.size.signed_int_min(), dest.layout)
        });
        this.write_immediate(*res, &dest)?;
    }
    // Fill remainder with zeros
    for i in op_len..dest_len {
        let dest = this.project_index(&dest, i)?;
        this.write_scalar(Scalar::from_int(0, dest.layout.size), &dest)?;
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

/// Conditionally multiplies the packed floating-point elements in
/// `left` and `right` using the high 4 bits in `imm`, sums the calculated
/// products (up to 4), and conditionally stores the sum in `dest` using
/// the low 4 bits of `imm`.
fn conditional_dot_product<'tcx>(
    this: &mut crate::MiriInterpCx<'_, 'tcx>,
    left: &OpTy<'tcx, Provenance>,
    right: &OpTy<'tcx, Provenance>,
    imm: &OpTy<'tcx, Provenance>,
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    let (left, left_len) = this.operand_to_simd(left)?;
    let (right, right_len) = this.operand_to_simd(right)?;
    let (dest, dest_len) = this.place_to_simd(dest)?;

    assert_eq!(left_len, right_len);
    assert!(dest_len <= 4);

    let imm = this.read_scalar(imm)?.to_u8()?;

    let element_layout = left.layout.field(this, 0);

    // Calculate dot product
    // Elements are floating point numbers, but we can use `from_int`
    // because the representation of 0.0 is all zero bits.
    let mut sum = ImmTy::from_int(0u8, element_layout);
    for i in 0..left_len {
        if imm & (1 << i.checked_add(4).unwrap()) != 0 {
            let left = this.read_immediate(&this.project_index(&left, i)?)?;
            let right = this.read_immediate(&this.project_index(&right, i)?)?;

            let mul = this.wrapping_binary_op(mir::BinOp::Mul, &left, &right)?;
            sum = this.wrapping_binary_op(mir::BinOp::Add, &sum, &mul)?;
        }
    }

    // Write to destination (conditioned to imm)
    for i in 0..dest_len {
        let dest = this.project_index(&dest, i)?;

        if imm & (1 << i) != 0 {
            this.write_immediate(*sum, &dest)?;
        } else {
            this.write_scalar(Scalar::from_int(0u8, element_layout.size), &dest)?;
        }
    }

    Ok(())
}

/// Calculates two booleans.
///
/// The first is true when all the bits of `op & mask` are zero.
/// The second is true when `(op & mask) == mask`
fn test_bits_masked<'tcx>(
    this: &crate::MiriInterpCx<'_, 'tcx>,
    op: &OpTy<'tcx, Provenance>,
    mask: &OpTy<'tcx, Provenance>,
) -> InterpResult<'tcx, (bool, bool)> {
    assert_eq!(op.layout, mask.layout);

    let (op, op_len) = this.operand_to_simd(op)?;
    let (mask, mask_len) = this.operand_to_simd(mask)?;

    assert_eq!(op_len, mask_len);

    let mut all_zero = true;
    let mut masked_set = true;
    for i in 0..op_len {
        let op = this.project_index(&op, i)?;
        let mask = this.project_index(&mask, i)?;

        let op = this.read_scalar(&op)?.to_uint(op.layout.size)?;
        let mask = this.read_scalar(&mask)?.to_uint(mask.layout.size)?;
        all_zero &= (op & mask) == 0;
        masked_set &= (op & mask) == mask;
    }

    Ok((all_zero, masked_set))
}
