use rustc_apfloat::ieee::{DoubleS, HalfS, IeeeFloat, Semantics, SingleS};
use rustc_apfloat::{self, Float, FloatConvert, Round};
use rustc_middle::mir;
use rustc_middle::ty::{self, FloatTy};

use self::math::{HostFloatOperation, HostUnaryFloatOp, IeeeExt, host_unary_float_op};
use super::check_intrinsic_arg_count;
use crate::*;

fn sqrt<'tcx, F: Float + FloatConvert<F> + Into<Scalar>>(
    this: &mut MiriInterpCx<'tcx>,
    args: &[OpTy<'tcx>],
    dest: &MPlaceTy<'tcx>,
) -> InterpResult<'tcx> {
    let [f] = check_intrinsic_arg_count(args)?;
    math::sqrt_op::<F>(this, f, dest)
}

/// Determine which float operation on which type this is.
fn is_host_unary_float_op(intrinsic_name: &str) -> Option<(FloatTy, HostUnaryFloatOp)> {
    let (op, ty) = intrinsic_name.rsplit_once('f')?;

    let float_ty = match ty {
        "16" => FloatTy::F16,
        "32" => FloatTy::F32,
        "64" => FloatTy::F64,
        "128" => FloatTy::F128,
        _ => return None,
    };

    let host_float_op = match op {
        "sin" => HostUnaryFloatOp::Sin,
        "cos" => HostUnaryFloatOp::Cos,
        "exp" => HostUnaryFloatOp::Exp,
        "exp2" => HostUnaryFloatOp::Exp2,
        "log" => HostUnaryFloatOp::Log,
        "log10" => HostUnaryFloatOp::Log10,
        "log2" => HostUnaryFloatOp::Log2,
        _ => return None,
    };

    Some((float_ty, host_float_op))
}

fn pow_intrinsic<'tcx, S: Semantics>(
    this: &mut MiriInterpCx<'tcx>,
    args: &[OpTy<'tcx>],
    dest: &MPlaceTy<'tcx>,
) -> InterpResult<'tcx, ()>
where
    IeeeFloat<S>: HostFloatOperation + IeeeExt + Float + Into<Scalar>,
{
    let [f1, f2] = check_intrinsic_arg_count(args)?;
    let f1: IeeeFloat<S> = this.read_scalar(f1)?.to_float()?;
    let f2: IeeeFloat<S> = this.read_scalar(f2)?.to_float()?;

    let res = math::fixed_float_value(this, "pow", &[f1, f2]).unwrap_or_else(|| {
        // Using host floats (but it's fine, this operation does not have guaranteed precision).
        let res = f1.host_powf(f2);

        // Apply a relative error of 4ULP to introduce some non-determinism
        // simulating imprecise implementations and optimizations.
        math::apply_random_float_error_ulp(this, res, 4)
    });
    let res = this.adjust_nan(res, &[f1, f2]);
    this.write_scalar(res, dest)?;
    interp_ok(())
}
fn powi_intrinsic<'tcx, S: Semantics>(
    this: &mut MiriInterpCx<'tcx>,
    args: &[OpTy<'tcx>],
    dest: &MPlaceTy<'tcx>,
) -> InterpResult<'tcx, ()>
where
    IeeeFloat<S>: HostFloatOperation + IeeeExt + Float + Into<Scalar>,
{
    let [f, i] = check_intrinsic_arg_count(args)?;
    let f: IeeeFloat<S> = this.read_scalar(f)?.to_float()?;
    let i = this.read_scalar(i)?.to_i32()?;

    let res = math::fixed_powi_value(this, f, i).unwrap_or_else(|| {
        // Using host floats (but it's fine, this operation does not have guaranteed precision).
        let res = f.host_powi(i);

        // Apply a relative error of 4ULP to introduce some non-determinism
        // simulating imprecise implementations and optimizations.
        math::apply_random_float_error_ulp(this, res, 4)
    });
    let res = this.adjust_nan(res, &[f]);
    this.write_scalar(res, dest)?;
    interp_ok(())
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_math_intrinsic(
        &mut self,
        intrinsic_name: &str,
        _generic_args: ty::GenericArgsRef<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();

        match intrinsic_name {
            // Operations we can do with soft-floats.
            "sqrtf16" => sqrt::<rustc_apfloat::ieee::Half>(this, args, dest)?,
            "sqrtf32" => sqrt::<rustc_apfloat::ieee::Single>(this, args, dest)?,
            "sqrtf64" => sqrt::<rustc_apfloat::ieee::Double>(this, args, dest)?,
            "sqrtf128" => sqrt::<rustc_apfloat::ieee::Quad>(this, args, dest)?,

            #[rustfmt::skip]
            | "fadd_fast"
            | "fsub_fast"
            | "fmul_fast"
            | "fdiv_fast"
            | "frem_fast"
            => {
                let [a, b] = check_intrinsic_arg_count(args)?;
                let a = this.read_immediate(a)?;
                let b = this.read_immediate(b)?;
                let op = match intrinsic_name {
                    "fadd_fast" => mir::BinOp::Add,
                    "fsub_fast" => mir::BinOp::Sub,
                    "fmul_fast" => mir::BinOp::Mul,
                    "fdiv_fast" => mir::BinOp::Div,
                    "frem_fast" => mir::BinOp::Rem,
                    _ => bug!(),
                };
                let float_finite = |x: &ImmTy<'tcx>| -> InterpResult<'tcx, bool> {
                    let ty::Float(fty) = x.layout.ty.kind() else {
                        bug!("float_finite: non-float input type {}", x.layout.ty)
                    };
                    interp_ok(match fty {
                        FloatTy::F16 => x.to_scalar().to_f16()?.is_finite(),
                        FloatTy::F32 => x.to_scalar().to_f32()?.is_finite(),
                        FloatTy::F64 => x.to_scalar().to_f64()?.is_finite(),
                        FloatTy::F128 => x.to_scalar().to_f128()?.is_finite(),
                    })
                };
                match (float_finite(&a)?, float_finite(&b)?) {
                    (false, false) => throw_ub_format!(
                        "`{intrinsic_name}` intrinsic called with non-finite value as both parameters",
                    ),
                    (false, _) => throw_ub_format!(
                        "`{intrinsic_name}` intrinsic called with non-finite value as first parameter",
                    ),
                    (_, false) => throw_ub_format!(
                        "`{intrinsic_name}` intrinsic called with non-finite value as second parameter",
                    ),
                    _ => {}
                }
                let res = this.binary_op(op, &a, &b)?;
                // This cannot be a NaN so we also don't have to apply any non-determinism.
                // (Also, `binary_op` already called `generate_nan` if needed.)
                if !float_finite(&res)? {
                    throw_ub_format!("`{intrinsic_name}` intrinsic produced non-finite value as result");
                }
                // Apply a relative error of 4ULP to simulate non-deterministic precision loss
                // due to optimizations.
                let res = math::apply_random_float_error_to_imm(this, res, 4)?;
                this.write_immediate(*res, dest)?;
            }

            "float_to_int_unchecked" => {
                let [val] = check_intrinsic_arg_count(args)?;
                let val = this.read_immediate(val)?;

                let res = this
                    .float_to_int_checked(&val, dest.layout, Round::TowardZero)?
                    .ok_or_else(|| {
                        err_ub_format!(
                            "`float_to_int_unchecked` intrinsic called on {val} which cannot be represented in target type `{:?}`",
                            dest.layout.ty
                        )
                    })?;

                this.write_immediate(*res, dest)?;
            }

            // Operations that need host floats.
            _ if let Some((float_ty, op)) = is_host_unary_float_op(intrinsic_name) => {
                let [f] = check_intrinsic_arg_count(args)?;
                match float_ty {
                    FloatTy::F16 => host_unary_float_op::<HalfS>(this, f, op, dest)?,
                    FloatTy::F32 => host_unary_float_op::<SingleS>(this, f, op, dest)?,
                    FloatTy::F64 => host_unary_float_op::<DoubleS>(this, f, op, dest)?,
                    FloatTy::F128 => todo!("f128"), // FIXME(f128)
                };
            }

            "powf16" => pow_intrinsic::<HalfS>(this, args, dest)?,
            "powf32" => pow_intrinsic::<SingleS>(this, args, dest)?,
            "powf64" => pow_intrinsic::<DoubleS>(this, args, dest)?,
            "powf128" => todo!("f128"), // FIXME(f128)

            "powif16" => powi_intrinsic::<HalfS>(this, args, dest)?,
            "powif32" => powi_intrinsic::<SingleS>(this, args, dest)?,
            "powif64" => powi_intrinsic::<DoubleS>(this, args, dest)?,
            "powif128" => todo!("f128"), // FIXME(f128)

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}

/// Compute a CRC32 checksum using the given polynomial.
///
/// `bit_size` is the number of relevant data bits (8, 16, 32, or 64).
/// Only the low `bit_size` bits of `data` are used; higher bits must be zero.
/// `polynomial` includes the leading 1 bit (e.g. `0x11EDC6F41` for CRC32C).
///
/// Following hardware CRC conventions, `crc` and `data` bits are assumed to be reversed,
/// and output bits will be equally reversed.
pub(crate) fn compute_crc32(crc: u32, data: u64, bit_size: u32, polynomial: u128) -> u32 {
    assert!(
        bit_size == 64 || data < 1u64.strict_shl(bit_size),
        "crc32: `data` is larger than {bit_size} bits"
    );
    // Bit-reverse inputs to match hardware CRC conventions.
    let crc = u128::from(crc.reverse_bits());
    // Reverse all 64 bits of `data`, then shift right by `64 - bit_size`. This
    // discards the (now-reversed) higher bits, leaving only the reversed low
    // `bit_size` bits in the lowest positions (with zeros above).
    let v = u128::from(data.reverse_bits() >> (64u32.strict_sub(bit_size)));

    // Perform polynomial division modulo 2.
    // The algorithm for the division is an adapted version of the
    // schoolbook division algorithm used for normal integer or polynomial
    // division. In this context, the quotient is not calculated, since
    // only the remainder is needed.
    //
    // The algorithm works as follows:
    // 1. Pull down digits until division can be performed. In the context of division
    //    modulo 2 it means locating the most significant digit of the dividend and shifting
    //    the divisor such that the position of the divisors most significand digit and the
    //    dividends most significand digit match.
    // 2. Perform a division and determine the remainder. Since it is arithmetic modulo 2,
    //    this operation is a simple bitwise exclusive or.
    // 3. Repeat steps 1. and 2. until the full remainder is calculated. This is the case
    //    once the degree of the remainder polynomial is smaller than the degree of the
    //    divisor polynomial. In other words, the number of leading zeros of the remainder
    //    is larger than the number of leading zeros of the divisor. It is important to
    //    note that standard arithmetic comparison is not applicable here:
    //    0b10011 / 0b11111 = 0b01100 is a valid division, even though the dividend is
    //    smaller than the divisor.
    let mut dividend = (crc << bit_size) ^ (v << 32);
    while dividend.leading_zeros() <= polynomial.leading_zeros() {
        dividend ^= (polynomial << polynomial.leading_zeros()) >> dividend.leading_zeros();
    }

    u32::try_from(dividend).unwrap().reverse_bits()
}
