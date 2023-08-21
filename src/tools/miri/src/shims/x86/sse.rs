use rustc_apfloat::{ieee::Single, Float as _};
use rustc_middle::mir;
use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

use rand::Rng as _;

use crate::*;
use shims::foreign_items::EmulateByNameResult;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn emulate_x86_sse_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, EmulateByNameResult<'mir, 'tcx>> {
        let this = self.eval_context_mut();
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.sse.").unwrap();
        // All these intrinsics operate on 128-bit (f32x4) SIMD vectors unless stated otherwise.
        // Many intrinsic names are sufixed with "ps" (packed single) or "ss" (scalar single),
        // where single means single precision floating point (f32). "ps" means thet the operation
        // is performed on each element of the vector, while "ss" means that the operation is
        // performed only on the first element, copying the remaining elements from the input
        // vector (for binary operations, from the left-hand side).
        match unprefixed_name {
            // Used to implement _mm_{add,sub,mul,div,min,max}_ss functions.
            // Performs the operations on the first component of `left` and
            // `right` and copies the remaining components from `left`.
            "add.ss" | "sub.ss" | "mul.ss" | "div.ss" | "min.ss" | "max.ss" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = match unprefixed_name {
                    "add.ss" => FloatBinOp::Arith(mir::BinOp::Add),
                    "sub.ss" => FloatBinOp::Arith(mir::BinOp::Sub),
                    "mul.ss" => FloatBinOp::Arith(mir::BinOp::Mul),
                    "div.ss" => FloatBinOp::Arith(mir::BinOp::Div),
                    "min.ss" => FloatBinOp::Min,
                    "max.ss" => FloatBinOp::Max,
                    _ => unreachable!(),
                };

                bin_op_ss(this, which, left, right, dest)?;
            }
            // Used to implement _mm_min_ps and _mm_max_ps functions.
            // Note that the semantics are a bit different from Rust simd_min
            // and simd_max intrinsics regarding handling of NaN and -0.0: Rust
            // matches the IEEE min/max operations, while x86 has different
            // semantics.
            "min.ps" | "max.ps" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = match unprefixed_name {
                    "min.ps" => FloatBinOp::Min,
                    "max.ps" => FloatBinOp::Max,
                    _ => unreachable!(),
                };

                bin_op_ps(this, which, left, right, dest)?;
            }
            // Used to implement _mm_{sqrt,rcp,rsqrt}_ss functions.
            // Performs the operations on the first component of `op` and
            // copies the remaining components from `op`.
            "sqrt.ss" | "rcp.ss" | "rsqrt.ss" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = match unprefixed_name {
                    "sqrt.ss" => FloatUnaryOp::Sqrt,
                    "rcp.ss" => FloatUnaryOp::Rcp,
                    "rsqrt.ss" => FloatUnaryOp::Rsqrt,
                    _ => unreachable!(),
                };

                unary_op_ss(this, which, op, dest)?;
            }
            // Used to implement _mm_{sqrt,rcp,rsqrt}_ss functions.
            // Performs the operations on all components of `op`.
            "sqrt.ps" | "rcp.ps" | "rsqrt.ps" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = match unprefixed_name {
                    "sqrt.ps" => FloatUnaryOp::Sqrt,
                    "rcp.ps" => FloatUnaryOp::Rcp,
                    "rsqrt.ps" => FloatUnaryOp::Rsqrt,
                    _ => unreachable!(),
                };

                unary_op_ps(this, which, op, dest)?;
            }
            // Used to implement the _mm_cmp_ss function.
            // Performs a comparison operation on the first component of `left`
            // and `right`, returning 0 if false or `u32::MAX` if true. The remaining
            // components are copied from `left`.
            "cmp.ss" => {
                let [left, right, imm] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = match this.read_scalar(imm)?.to_i8()? {
                    0 => FloatBinOp::Cmp(FloatCmpOp::Eq),
                    1 => FloatBinOp::Cmp(FloatCmpOp::Lt),
                    2 => FloatBinOp::Cmp(FloatCmpOp::Le),
                    3 => FloatBinOp::Cmp(FloatCmpOp::Unord),
                    4 => FloatBinOp::Cmp(FloatCmpOp::Neq),
                    5 => FloatBinOp::Cmp(FloatCmpOp::Nlt),
                    6 => FloatBinOp::Cmp(FloatCmpOp::Nle),
                    7 => FloatBinOp::Cmp(FloatCmpOp::Ord),
                    imm => {
                        throw_unsup_format!(
                            "invalid 3rd parameter of llvm.x86.sse.cmp.ps: {}",
                            imm
                        );
                    }
                };

                bin_op_ss(this, which, left, right, dest)?;
            }
            // Used to implement the _mm_cmp_ps function.
            // Performs a comparison operation on each component of `left`
            // and `right`. For each component, returns 0 if false or u32::MAX
            // if true.
            "cmp.ps" => {
                let [left, right, imm] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let which = match this.read_scalar(imm)?.to_i8()? {
                    0 => FloatBinOp::Cmp(FloatCmpOp::Eq),
                    1 => FloatBinOp::Cmp(FloatCmpOp::Lt),
                    2 => FloatBinOp::Cmp(FloatCmpOp::Le),
                    3 => FloatBinOp::Cmp(FloatCmpOp::Unord),
                    4 => FloatBinOp::Cmp(FloatCmpOp::Neq),
                    5 => FloatBinOp::Cmp(FloatCmpOp::Nlt),
                    6 => FloatBinOp::Cmp(FloatCmpOp::Nle),
                    7 => FloatBinOp::Cmp(FloatCmpOp::Ord),
                    imm => {
                        throw_unsup_format!(
                            "invalid 3rd parameter of llvm.x86.sse.cmp.ps: {}",
                            imm
                        );
                    }
                };

                bin_op_ps(this, which, left, right, dest)?;
            }
            // Used to implement _mm_{,u}comi{eq,lt,le,gt,ge,neq}_ps functions.
            // Compares the first component of `left` and `right` and returns
            // a scalar value (0 or 1).
            "comieq.ss" | "comilt.ss" | "comile.ss" | "comigt.ss" | "comige.ss" | "comineq.ss"
            | "ucomieq.ss" | "ucomilt.ss" | "ucomile.ss" | "ucomigt.ss" | "ucomige.ss"
            | "ucomineq.ss" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;

                assert_eq!(left_len, right_len);

                let left = this.read_scalar(&this.project_index(&left, 0)?)?.to_f32()?;
                let right = this.read_scalar(&this.project_index(&right, 0)?)?.to_f32()?;
                // The difference between the com* and *ucom variants is signaling
                // of exceptions when either argument is a quiet NaN. We do not
                // support accessing the SSE status register from miri (or from Rust,
                // for that matter), so we treat equally both variants.
                let res = match unprefixed_name {
                    "comieq.ss" | "ucomieq.ss" => left == right,
                    "comilt.ss" | "ucomilt.ss" => left < right,
                    "comile.ss" | "ucomile.ss" => left <= right,
                    "comigt.ss" | "ucomigt.ss" => left > right,
                    "comige.ss" | "ucomige.ss" => left >= right,
                    "comineq.ss" | "ucomineq.ss" => left != right,
                    _ => unreachable!(),
                };
                this.write_scalar(Scalar::from_i32(i32::from(res)), dest)?;
            }
            // Use to implement _mm_cvtss_si32 and _mm_cvttss_si32.
            // Converts the first component of `op` from f32 to i32.
            "cvtss2si" | "cvttss2si" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let (op, _) = this.operand_to_simd(op)?;

                let op = this.read_scalar(&this.project_index(&op, 0)?)?.to_f32()?;

                let rnd = match unprefixed_name {
                    // "current SSE rounding mode", assume nearest
                    // https://www.felixcloutier.com/x86/cvtss2si
                    "cvtss2si" => rustc_apfloat::Round::NearestTiesToEven,
                    // always truncate
                    // https://www.felixcloutier.com/x86/cvttss2si
                    "cvttss2si" => rustc_apfloat::Round::TowardZero,
                    _ => unreachable!(),
                };

                let res = this.float_to_int_checked(op, dest.layout.ty, rnd).unwrap_or_else(|| {
                    // Fallback to minimum acording to SSE semantics.
                    Scalar::from_i32(i32::MIN)
                });

                this.write_scalar(res, dest)?;
            }
            // Use to implement _mm_cvtss_si64 and _mm_cvttss_si64.
            // Converts the first component of `op` from f32 to i64.
            "cvtss2si64" | "cvttss2si64" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let (op, _) = this.operand_to_simd(op)?;

                let op = this.read_scalar(&this.project_index(&op, 0)?)?.to_f32()?;

                let rnd = match unprefixed_name {
                    // "current SSE rounding mode", assume nearest
                    // https://www.felixcloutier.com/x86/cvtss2si
                    "cvtss2si64" => rustc_apfloat::Round::NearestTiesToEven,
                    // always truncate
                    // https://www.felixcloutier.com/x86/cvttss2si
                    "cvttss2si64" => rustc_apfloat::Round::TowardZero,
                    _ => unreachable!(),
                };

                let res = this.float_to_int_checked(op, dest.layout.ty, rnd).unwrap_or_else(|| {
                    // Fallback to minimum acording to SSE semantics.
                    Scalar::from_i64(i64::MIN)
                });

                this.write_scalar(res, dest)?;
            }
            // Used to implement the _mm_cvtsi32_ss function.
            // Converts `right` from i32 to f32. Returns a SIMD vector with
            // the result in the first component and the remaining components
            // are copied from `left`.
            // https://www.felixcloutier.com/x86/cvtsi2ss
            "cvtsi2ss" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, left_len);

                let right = this.read_scalar(right)?.to_i32()?;

                let res0 = Scalar::from_f32(Single::from_i128(right.into()).value);
                this.write_scalar(res0, &this.project_index(&dest, 0)?)?;

                for i in 1..dest_len {
                    let left = this.read_immediate(&this.project_index(&left, i)?)?;
                    let dest = this.project_index(&dest, i)?;

                    this.write_immediate(*left, &dest)?;
                }
            }
            // Used to implement the _mm_cvtsi64_ss function.
            // Converts `right` from i64 to f32. Returns a SIMD vector with
            // the result in the first component and the remaining components
            // are copied from `left`.
            // https://www.felixcloutier.com/x86/cvtsi2ss
            "cvtsi642ss" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

                let (left, left_len) = this.operand_to_simd(left)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, left_len);

                let right = this.read_scalar(right)?.to_i64()?;

                let res0 = Scalar::from_f32(Single::from_i128(right.into()).value);
                this.write_scalar(res0, &this.project_index(&dest, 0)?)?;

                for i in 1..dest_len {
                    let left = this.read_immediate(&this.project_index(&left, i)?)?;
                    let dest = this.project_index(&dest, i)?;

                    this.write_immediate(*left, &dest)?;
                }
            }
            // Used to implement the _mm_movemask_ps function.
            // Returns a scalar integer where the i-th bit is the highest
            // bit of the i-th component of `op`.
            // https://www.felixcloutier.com/x86/movmskps
            "movmsk.ps" => {
                let [op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let (op, op_len) = this.operand_to_simd(op)?;

                let mut res = 0;
                for i in 0..op_len {
                    let op = this.read_scalar(&this.project_index(&op, i)?)?;
                    let op = op.to_u32()?;

                    res |= (op >> 31) << i;
                }

                this.write_scalar(Scalar::from_u32(res), dest)?;
            }
            _ => return Ok(EmulateByNameResult::NotSupported),
        }
        Ok(EmulateByNameResult::NeedsJumping)
    }
}

/// Floating point comparison operation
///
/// <https://www.felixcloutier.com/x86/cmpss>
/// <https://www.felixcloutier.com/x86/cmpps>
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
    Min,
    /// Maximum value (with SSE semantics)
    ///
    /// <https://www.felixcloutier.com/x86/maxss>
    /// <https://www.felixcloutier.com/x86/maxps>
    Max,
}

/// Performs `which` scalar operation on `left` and `right` and returns
/// the result.
fn bin_op_f32<'tcx>(
    this: &crate::MiriInterpCx<'_, 'tcx>,
    which: FloatBinOp,
    left: &ImmTy<'tcx, Provenance>,
    right: &ImmTy<'tcx, Provenance>,
) -> InterpResult<'tcx, Scalar<Provenance>> {
    match which {
        FloatBinOp::Arith(which) => {
            let (res, _, _) = this.overflowing_binary_op(which, left, right)?;
            Ok(res)
        }
        FloatBinOp::Cmp(which) => {
            let left = left.to_scalar().to_f32()?;
            let right = right.to_scalar().to_f32()?;
            // FIXME: Make sure that these operations match the semantics of cmpps
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
            Ok(Scalar::from_u32(if res { u32::MAX } else { 0 }))
        }
        FloatBinOp::Min => {
            let left = left.to_scalar().to_f32()?;
            let right = right.to_scalar().to_f32()?;
            // SSE semantics to handle zero and NaN. Note that `x == Single::ZERO`
            // is true when `x` is either +0 or -0.
            if (left == Single::ZERO && right == Single::ZERO)
                || left.is_nan()
                || right.is_nan()
                || left >= right
            {
                Ok(Scalar::from_f32(right))
            } else {
                Ok(Scalar::from_f32(left))
            }
        }
        FloatBinOp::Max => {
            let left = left.to_scalar().to_f32()?;
            let right = right.to_scalar().to_f32()?;
            // SSE semantics to handle zero and NaN. Note that `x == Single::ZERO`
            // is true when `x` is either +0 or -0.
            if (left == Single::ZERO && right == Single::ZERO)
                || left.is_nan()
                || right.is_nan()
                || left <= right
            {
                Ok(Scalar::from_f32(right))
            } else {
                Ok(Scalar::from_f32(left))
            }
        }
    }
}

/// Performs `which` operation on the first component of `left` and `right`
/// and copies the other components from `left`. The result is stored in `dest`.
fn bin_op_ss<'tcx>(
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

    let res0 = bin_op_f32(
        this,
        which,
        &this.read_immediate(&this.project_index(&left, 0)?)?,
        &this.read_immediate(&this.project_index(&right, 0)?)?,
    )?;
    this.write_scalar(res0, &this.project_index(&dest, 0)?)?;

    for i in 1..dest_len {
        let left = this.read_immediate(&this.project_index(&left, i)?)?;
        let dest = this.project_index(&dest, i)?;

        this.write_immediate(*left, &dest)?;
    }

    Ok(())
}

/// Performs `which` operation on each component of `left`, and
/// `right` storing the result is stored in `dest`.
fn bin_op_ps<'tcx>(
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

        let res = bin_op_f32(this, which, &left, &right)?;
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
        let op = this.read_immediate(&this.project_index(&op, i)?)?;
        let dest = this.project_index(&dest, i)?;

        this.write_immediate(*op, &dest)?;
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
