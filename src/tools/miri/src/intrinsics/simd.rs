use rustc_apfloat::ieee::{DoubleS, HalfS, IeeeFloat, QuadS, SingleS};
use rustc_middle::ty;
use rustc_middle::ty::FloatTy;

use super::check_intrinsic_arg_count;
use crate::math::{HostUnaryFloatOp, host_unary_float_op};
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
            // Operations we can do with soft-floats.
            "fsqrt" => {
                let [op] = check_intrinsic_arg_count(args)?;
                let (op, op_len) = this.project_to_simd(op)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, op_len);

                for i in 0..dest_len {
                    let op = this.project_index(&op, i)?;
                    let dest = this.project_index(&dest, i)?;
                    let ty::Float(float_ty) = op.layout.ty.kind() else {
                        span_bug!(this.cur_span(), "{} operand is not a float", intrinsic_name)
                    };
                    match float_ty {
                        FloatTy::F16 => math::sqrt_op::<IeeeFloat<HalfS>>(this, &op, &dest)?,
                        FloatTy::F32 => math::sqrt_op::<IeeeFloat<SingleS>>(this, &op, &dest)?,
                        FloatTy::F64 => math::sqrt_op::<IeeeFloat<DoubleS>>(this, &op, &dest)?,
                        FloatTy::F128 => math::sqrt_op::<IeeeFloat<QuadS>>(this, &op, &dest)?,
                    };
                }
            }

            // Operations that need host floats.
            "fsin" | "fcos" | "fexp" | "fexp2" | "flog" | "flog2" | "flog10" => {
                let [op] = check_intrinsic_arg_count(args)?;
                let (op, op_len) = this.project_to_simd(op)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, op_len);

                let host_op = match intrinsic_name {
                    "fsin" => HostUnaryFloatOp::Sin,
                    "fcos" => HostUnaryFloatOp::Cos,
                    "fexp" => HostUnaryFloatOp::Exp,
                    "fexp2" => HostUnaryFloatOp::Exp2,
                    "flog" => HostUnaryFloatOp::Log,
                    "flog2" => HostUnaryFloatOp::Log2,
                    "flog10" => HostUnaryFloatOp::Log10,
                    _ => bug!(),
                };

                for i in 0..dest_len {
                    let op = this.project_index(&op, i)?;
                    let dest = this.project_index(&dest, i)?;
                    let ty::Float(float_ty) = op.layout.ty.kind() else {
                        span_bug!(this.cur_span(), "{} operand is not a float", intrinsic_name)
                    };
                    // Using host floats except for sqrt (but it's fine, these operations do not
                    // have guaranteed precision).
                    match float_ty {
                        FloatTy::F16 => host_unary_float_op::<HalfS>(this, &op, host_op, &dest)?,
                        FloatTy::F32 => host_unary_float_op::<SingleS>(this, &op, host_op, &dest)?,
                        FloatTy::F64 => host_unary_float_op::<DoubleS>(this, &op, host_op, &dest)?,
                        FloatTy::F128 => unimplemented!("f128"), // FIXME(f128)
                    }
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
