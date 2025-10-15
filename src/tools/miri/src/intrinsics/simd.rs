use rand::Rng;
use rustc_apfloat::Float;
use rustc_middle::ty;
use rustc_middle::ty::FloatTy;

use super::check_intrinsic_arg_count;
use crate::helpers::{ToHost, ToSoft};
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Calls the simd intrinsic `intrinsic`; the `simd_` prefix has already been removed.
    /// Returns `Ok(true)` if the intrinsic was handled.
    fn emulate_simd_intrinsic(
        &mut self,
        intrinsic_name: &str,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        match intrinsic_name {
            #[rustfmt::skip]
            | "fsqrt"
            | "fsin"
            | "fcos"
            | "fexp"
            | "fexp2"
            | "flog"
            | "flog2"
            | "flog10"
            => {
                let [op] = check_intrinsic_arg_count(args)?;
                let (op, op_len) = this.project_to_simd(op)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, op_len);

                for i in 0..dest_len {
                    let op = this.read_immediate(&this.project_index(&op, i)?)?;
                    let dest = this.project_index(&dest, i)?;
                            let ty::Float(float_ty) = op.layout.ty.kind() else {
                                span_bug!(this.cur_span(), "{} operand is not a float", intrinsic_name)
                            };
                            // Using host floats except for sqrt (but it's fine, these operations do not
                            // have guaranteed precision).
                    let val = match float_ty {
                                FloatTy::F16 => unimplemented!("f16_f128"),
                                FloatTy::F32 => {
                                    let f = op.to_scalar().to_f32()?;
                            let res = match intrinsic_name {
                                        "fsqrt" => math::sqrt(f),
                                        "fsin" => f.to_host().sin().to_soft(),
                                        "fcos" => f.to_host().cos().to_soft(),
                                        "fexp" => f.to_host().exp().to_soft(),
                                        "fexp2" => f.to_host().exp2().to_soft(),
                                        "flog" => f.to_host().ln().to_soft(),
                                        "flog2" => f.to_host().log2().to_soft(),
                                        "flog10" => f.to_host().log10().to_soft(),
                                        _ => bug!(),
                                    };
                                    let res = this.adjust_nan(res, &[f]);
                                    Scalar::from(res)
                                }
                                FloatTy::F64 => {
                                    let f = op.to_scalar().to_f64()?;
                            let res = match intrinsic_name {
                                        "fsqrt" => math::sqrt(f),
                                        "fsin" => f.to_host().sin().to_soft(),
                                        "fcos" => f.to_host().cos().to_soft(),
                                        "fexp" => f.to_host().exp().to_soft(),
                                        "fexp2" => f.to_host().exp2().to_soft(),
                                        "flog" => f.to_host().ln().to_soft(),
                                        "flog2" => f.to_host().log2().to_soft(),
                                        "flog10" => f.to_host().log10().to_soft(),
                                        _ => bug!(),
                                    };
                                    let res = this.adjust_nan(res, &[f]);
                                    Scalar::from(res)
                                }
                                FloatTy::F128 => unimplemented!("f16_f128"),
                            };

                    this.write_scalar(val, &dest)?;
                }
            }
            "fma" | "relaxed_fma" => {
                let [a, b, c] = check_intrinsic_arg_count(args)?;
                let (a, a_len) = this.project_to_simd(a)?;
                let (b, b_len) = this.project_to_simd(b)?;
                let (c, c_len) = this.project_to_simd(c)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, a_len);
                assert_eq!(dest_len, b_len);
                assert_eq!(dest_len, c_len);

                for i in 0..dest_len {
                    let a = this.read_scalar(&this.project_index(&a, i)?)?;
                    let b = this.read_scalar(&this.project_index(&b, i)?)?;
                    let c = this.read_scalar(&this.project_index(&c, i)?)?;
                    let dest = this.project_index(&dest, i)?;

                    let fuse: bool = intrinsic_name == "fma"
                        || (this.machine.float_nondet && this.machine.rng.get_mut().random());

                    // Works for f32 and f64.
                    // FIXME: using host floats to work around https://github.com/rust-lang/miri/issues/2468.
                    let ty::Float(float_ty) = dest.layout.ty.kind() else {
                        span_bug!(this.cur_span(), "{} operand is not a float", intrinsic_name)
                    };
                    let val = match float_ty {
                        FloatTy::F16 => unimplemented!("f16_f128"),
                        FloatTy::F32 => {
                            let a = a.to_f32()?;
                            let b = b.to_f32()?;
                            let c = c.to_f32()?;
                            let res = if fuse {
                                a.mul_add(b, c).value
                            } else {
                                ((a * b).value + c).value
                            };
                            let res = this.adjust_nan(res, &[a, b, c]);
                            Scalar::from(res)
                        }
                        FloatTy::F64 => {
                            let a = a.to_f64()?;
                            let b = b.to_f64()?;
                            let c = c.to_f64()?;
                            let res = if fuse {
                                a.mul_add(b, c).value
                            } else {
                                ((a * b).value + c).value
                            };
                            let res = this.adjust_nan(res, &[a, b, c]);
                            Scalar::from(res)
                        }
                        FloatTy::F128 => unimplemented!("f16_f128"),
                    };
                    this.write_scalar(val, &dest)?;
                }
            }
            "expose_provenance" => {
                let [op] = check_intrinsic_arg_count(args)?;
                let (op, op_len) = this.project_to_simd(op)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, op_len);

                for i in 0..dest_len {
                    let op = this.read_immediate(&this.project_index(&op, i)?)?;
                    let dest = this.project_index(&dest, i)?;

                    let val = match (op.layout.ty.kind(), dest.layout.ty.kind()) {
                        // Ptr/Int casts
                        (ty::RawPtr(..), ty::Int(_) | ty::Uint(_)) =>
                            this.pointer_expose_provenance_cast(&op, dest.layout)?,
                        // Error otherwise
                        _ =>
                            throw_unsup_format!(
                                "Unsupported `simd_expose_provenance` from element type {from_ty} to {to_ty}",
                                from_ty = op.layout.ty,
                                to_ty = dest.layout.ty,
                            ),
                    };
                    this.write_immediate(*val, &dest)?;
                }
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
