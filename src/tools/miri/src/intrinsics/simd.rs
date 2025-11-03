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
