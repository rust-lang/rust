use rustc_apfloat::{self, Float, FloatConvert, Round};
use rustc_middle::mir;
use rustc_middle::ty::{self, FloatTy};

use self::helpers::{ToHost, ToSoft};
use super::check_intrinsic_arg_count;
use crate::*;

fn sqrt<'tcx, F: Float + FloatConvert<F> + Into<Scalar>>(
    this: &mut MiriInterpCx<'tcx>,
    args: &[OpTy<'tcx>],
    dest: &MPlaceTy<'tcx>,
) -> InterpResult<'tcx> {
    let [f] = check_intrinsic_arg_count(args)?;
    let f = this.read_scalar(f)?;
    let f: F = f.to_float()?;
    // Sqrt is specified to be fully precise.
    let res = math::sqrt(f);
    let res = this.adjust_nan(res, &[f]);
    this.write_scalar(res, dest)
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
            #[rustfmt::skip]
            | "sinf32"
            | "cosf32"
            | "expf32"
            | "exp2f32"
            | "logf32"
            | "log10f32"
            | "log2f32"
            => {
                let [f] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f32()?;

                let res = math::fixed_float_value(this, intrinsic_name, &[f]).unwrap_or_else(|| {
                    // Using host floats (but it's fine, these operations do not have
                    // guaranteed precision).
                    let host = f.to_host();
                    let res = match intrinsic_name {
                        "sinf32" => host.sin(),
                        "cosf32" => host.cos(),
                        "expf32" => host.exp(),
                        "exp2f32" => host.exp2(),
                        "logf32" => host.ln(),
                        "log10f32" => host.log10(),
                        "log2f32" => host.log2(),
                        _ => bug!(),
                    };
                    let res = res.to_soft();

                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    let res = math::apply_random_float_error_ulp(
                        this,
                        res,
                        4,
                    );

                    // Clamp the result to the guaranteed range of this function according to the C standard,
                    // if any.
                    math::clamp_float_value(intrinsic_name, res)
                });
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }

            #[rustfmt::skip]
            | "sinf64"
            | "cosf64"
            | "expf64"
            | "exp2f64"
            | "logf64"
            | "log10f64"
            | "log2f64"
            => {
                let [f] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f64()?;

                let res = math::fixed_float_value(this, intrinsic_name, &[f]).unwrap_or_else(|| {
                    // Using host floats (but it's fine, these operations do not have
                    // guaranteed precision).
                    let host = f.to_host();
                    let res = match intrinsic_name {
                        "sinf64" => host.sin(),
                        "cosf64" => host.cos(),
                        "expf64" => host.exp(),
                        "exp2f64" => host.exp2(),
                        "logf64" => host.ln(),
                        "log10f64" => host.log10(),
                        "log2f64" => host.log2(),
                        _ => bug!(),
                    };
                    let res = res.to_soft();

                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    let res = math::apply_random_float_error_ulp(
                        this,
                        res,
                        4,
                    );

                    // Clamp the result to the guaranteed range of this function according to the C standard,
                    // if any.
                    math::clamp_float_value(intrinsic_name, res)
                });
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }

            "powf32" => {
                let [f1, f2] = check_intrinsic_arg_count(args)?;
                let f1 = this.read_scalar(f1)?.to_f32()?;
                let f2 = this.read_scalar(f2)?.to_f32()?;

                let res =
                    math::fixed_float_value(this, intrinsic_name, &[f1, f2]).unwrap_or_else(|| {
                        // Using host floats (but it's fine, this operation does not have guaranteed precision).
                        let res = f1.to_host().powf(f2.to_host()).to_soft();

                        // Apply a relative error of 4ULP to introduce some non-determinism
                        // simulating imprecise implementations and optimizations.
                        math::apply_random_float_error_ulp(this, res, 4)
                    });
                let res = this.adjust_nan(res, &[f1, f2]);
                this.write_scalar(res, dest)?;
            }
            "powf64" => {
                let [f1, f2] = check_intrinsic_arg_count(args)?;
                let f1 = this.read_scalar(f1)?.to_f64()?;
                let f2 = this.read_scalar(f2)?.to_f64()?;

                let res =
                    math::fixed_float_value(this, intrinsic_name, &[f1, f2]).unwrap_or_else(|| {
                        // Using host floats (but it's fine, this operation does not have guaranteed precision).
                        let res = f1.to_host().powf(f2.to_host()).to_soft();

                        // Apply a relative error of 4ULP to introduce some non-determinism
                        // simulating imprecise implementations and optimizations.
                        math::apply_random_float_error_ulp(this, res, 4)
                    });
                let res = this.adjust_nan(res, &[f1, f2]);
                this.write_scalar(res, dest)?;
            }

            "powif32" => {
                let [f, i] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f32()?;
                let i = this.read_scalar(i)?.to_i32()?;

                let res = math::fixed_powi_value(this, f, i).unwrap_or_else(|| {
                    // Using host floats (but it's fine, this operation does not have guaranteed precision).
                    let res = f.to_host().powi(i).to_soft();

                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    math::apply_random_float_error_ulp(this, res, 4)
                });
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }
            "powif64" => {
                let [f, i] = check_intrinsic_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f64()?;
                let i = this.read_scalar(i)?.to_i32()?;

                let res = math::fixed_powi_value(this, f, i).unwrap_or_else(|| {
                    // Using host floats (but it's fine, this operation does not have guaranteed precision).
                    let res = f.to_host().powi(i).to_soft();

                    // Apply a relative error of 4ULP to introduce some non-determinism
                    // simulating imprecise implementations and optimizations.
                    math::apply_random_float_error_ulp(this, res, 4)
                });
                let res = this.adjust_nan(res, &[f]);
                this.write_scalar(res, dest)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
