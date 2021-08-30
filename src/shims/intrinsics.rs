use std::iter;

use log::trace;

use rustc_apfloat::{Float, Round};
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::{mir, mir::BinOp, ty, ty::FloatTy};
use rustc_target::abi::{Align, Integer, LayoutOf};

use crate::*;
use helpers::check_arg_count;

pub enum AtomicOp {
    MirOp(mir::BinOp, bool),
    Max,
    Min,
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn call_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(&PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
        _unwind: StackPopUnwind,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        if this.emulate_intrinsic(instance, args, ret)? {
            return Ok(());
        }

        // All supported intrinsics have a return place.
        let intrinsic_name = &*this.tcx.item_name(instance.def_id()).as_str();
        let (dest, ret) = match ret {
            None => throw_unsup_format!("unimplemented (diverging) intrinsic: {}", intrinsic_name),
            Some(p) => p,
        };

        // Then handle terminating intrinsics.
        match intrinsic_name {
            // Miri overwriting CTFE intrinsics.
            "ptr_guaranteed_eq" => {
                let &[ref left, ref right] = check_arg_count(args)?;
                let left = this.read_immediate(left)?;
                let right = this.read_immediate(right)?;
                this.binop_ignore_overflow(mir::BinOp::Eq, &left, &right, dest)?;
            }
            "ptr_guaranteed_ne" => {
                let &[ref left, ref right] = check_arg_count(args)?;
                let left = this.read_immediate(left)?;
                let right = this.read_immediate(right)?;
                this.binop_ignore_overflow(mir::BinOp::Ne, &left, &right, dest)?;
            }

            // Raw memory accesses
            "volatile_load" => {
                let &[ref place] = check_arg_count(args)?;
                let place = this.deref_operand(place)?;
                this.copy_op(&place.into(), dest)?;
            }
            "volatile_store" => {
                let &[ref place, ref dest] = check_arg_count(args)?;
                let place = this.deref_operand(place)?;
                this.copy_op(dest, &place.into())?;
            }

            "write_bytes" | "volatile_set_memory" => {
                let &[ref ptr, ref val_byte, ref count] = check_arg_count(args)?;
                let ty = instance.substs.type_at(0);
                let ty_layout = this.layout_of(ty)?;
                let val_byte = this.read_scalar(val_byte)?.to_u8()?;
                let ptr = this.read_pointer(ptr)?;
                let count = this.read_scalar(count)?.to_machine_usize(this)?;
                let byte_count = ty_layout.size.checked_mul(count, this).ok_or_else(|| {
                    err_ub_format!("overflow computing total size of `{}`", intrinsic_name)
                })?;
                this.memory
                    .write_bytes(ptr, iter::repeat(val_byte).take(byte_count.bytes() as usize))?;
            }

            // Floating-point operations
            #[rustfmt::skip]
            | "sinf32"
            | "fabsf32"
            | "cosf32"
            | "sqrtf32"
            | "expf32"
            | "exp2f32"
            | "logf32"
            | "log10f32"
            | "log2f32"
            | "floorf32"
            | "ceilf32"
            | "truncf32"
            | "roundf32"
            => {
                let &[ref f] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(f)?.to_u32()?);
                let f = match intrinsic_name {
                    "sinf32" => f.sin(),
                    "fabsf32" => f.abs(),
                    "cosf32" => f.cos(),
                    "sqrtf32" => f.sqrt(),
                    "expf32" => f.exp(),
                    "exp2f32" => f.exp2(),
                    "logf32" => f.ln(),
                    "log10f32" => f.log10(),
                    "log2f32" => f.log2(),
                    "floorf32" => f.floor(),
                    "ceilf32" => f.ceil(),
                    "truncf32" => f.trunc(),
                    "roundf32" => f.round(),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u32(f.to_bits()), dest)?;
            }

            #[rustfmt::skip]
            | "sinf64"
            | "fabsf64"
            | "cosf64"
            | "sqrtf64"
            | "expf64"
            | "exp2f64"
            | "logf64"
            | "log10f64"
            | "log2f64"
            | "floorf64"
            | "ceilf64"
            | "truncf64"
            | "roundf64"
            => {
                let &[ref f] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(f)?.to_u64()?);
                let f = match intrinsic_name {
                    "sinf64" => f.sin(),
                    "fabsf64" => f.abs(),
                    "cosf64" => f.cos(),
                    "sqrtf64" => f.sqrt(),
                    "expf64" => f.exp(),
                    "exp2f64" => f.exp2(),
                    "logf64" => f.ln(),
                    "log10f64" => f.log10(),
                    "log2f64" => f.log2(),
                    "floorf64" => f.floor(),
                    "ceilf64" => f.ceil(),
                    "truncf64" => f.trunc(),
                    "roundf64" => f.round(),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u64(f.to_bits()), dest)?;
            }

            #[rustfmt::skip]
            | "fadd_fast"
            | "fsub_fast"
            | "fmul_fast"
            | "fdiv_fast"
            | "frem_fast"
            => {
                let &[ref a, ref b] = check_arg_count(args)?;
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
                let float_finite = |x: ImmTy<'tcx, _>| -> InterpResult<'tcx, bool> {
                    Ok(match x.layout.ty.kind() {
                        ty::Float(FloatTy::F32) => x.to_scalar()?.to_f32()?.is_finite(),
                        ty::Float(FloatTy::F64) => x.to_scalar()?.to_f64()?.is_finite(),
                        _ => bug!(
                            "`{}` called with non-float input type {:?}",
                            intrinsic_name,
                            x.layout.ty
                        ),
                    })
                };
                match (float_finite(a)?, float_finite(b)?) {
                    (false, false) => throw_ub_format!(
                        "`{}` intrinsic called with non-finite value as both parameters",
                        intrinsic_name,
                    ),
                    (false, _) => throw_ub_format!(
                        "`{}` intrinsic called with non-finite value as first parameter",
                        intrinsic_name,
                    ),
                    (_, false) => throw_ub_format!(
                        "`{}` intrinsic called with non-finite value as second parameter",
                        intrinsic_name,
                    ),
                    _ => {}
                }
                this.binop_ignore_overflow(op, &a, &b, dest)?;
            }

            #[rustfmt::skip]
            | "minnumf32"
            | "maxnumf32"
            | "copysignf32"
            => {
                let &[ref a, ref b] = check_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f32()?;
                let b = this.read_scalar(b)?.to_f32()?;
                let res = match intrinsic_name {
                    "minnumf32" => a.min(b),
                    "maxnumf32" => a.max(b),
                    "copysignf32" => a.copy_sign(b),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_f32(res), dest)?;
            }

            #[rustfmt::skip]
            | "minnumf64"
            | "maxnumf64"
            | "copysignf64"
            => {
                let &[ref a, ref b] = check_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f64()?;
                let b = this.read_scalar(b)?.to_f64()?;
                let res = match intrinsic_name {
                    "minnumf64" => a.min(b),
                    "maxnumf64" => a.max(b),
                    "copysignf64" => a.copy_sign(b),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_f64(res), dest)?;
            }

            "powf32" => {
                let &[ref f, ref f2] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(f)?.to_u32()?);
                let f2 = f32::from_bits(this.read_scalar(f2)?.to_u32()?);
                this.write_scalar(Scalar::from_u32(f.powf(f2).to_bits()), dest)?;
            }

            "powf64" => {
                let &[ref f, ref f2] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(f)?.to_u64()?);
                let f2 = f64::from_bits(this.read_scalar(f2)?.to_u64()?);
                this.write_scalar(Scalar::from_u64(f.powf(f2).to_bits()), dest)?;
            }

            "fmaf32" => {
                let &[ref a, ref b, ref c] = check_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f32()?;
                let b = this.read_scalar(b)?.to_f32()?;
                let c = this.read_scalar(c)?.to_f32()?;
                let res = a.mul_add(b, c).value;
                this.write_scalar(Scalar::from_f32(res), dest)?;
            }

            "fmaf64" => {
                let &[ref a, ref b, ref c] = check_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f64()?;
                let b = this.read_scalar(b)?.to_f64()?;
                let c = this.read_scalar(c)?.to_f64()?;
                let res = a.mul_add(b, c).value;
                this.write_scalar(Scalar::from_f64(res), dest)?;
            }

            "powif32" => {
                let &[ref f, ref i] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(f)?.to_u32()?);
                let i = this.read_scalar(i)?.to_i32()?;
                this.write_scalar(Scalar::from_u32(f.powi(i).to_bits()), dest)?;
            }

            "powif64" => {
                let &[ref f, ref i] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(f)?.to_u64()?);
                let i = this.read_scalar(i)?.to_i32()?;
                this.write_scalar(Scalar::from_u64(f.powi(i).to_bits()), dest)?;
            }

            "float_to_int_unchecked" => {
                let &[ref val] = check_arg_count(args)?;
                let val = this.read_immediate(val)?;

                let res = match val.layout.ty.kind() {
                    ty::Float(FloatTy::F32) =>
                        this.float_to_int_unchecked(val.to_scalar()?.to_f32()?, dest.layout.ty)?,
                    ty::Float(FloatTy::F64) =>
                        this.float_to_int_unchecked(val.to_scalar()?.to_f64()?, dest.layout.ty)?,
                    _ =>
                        bug!(
                            "`float_to_int_unchecked` called with non-float input type {:?}",
                            val.layout.ty
                        ),
                };

                this.write_scalar(res, dest)?;
            }

            // Atomic operations
            "atomic_load" => this.atomic_load(args, dest, AtomicReadOp::SeqCst)?,
            "atomic_load_relaxed" => this.atomic_load(args, dest, AtomicReadOp::Relaxed)?,
            "atomic_load_acq" => this.atomic_load(args, dest, AtomicReadOp::Acquire)?,

            "atomic_store" => this.atomic_store(args, AtomicWriteOp::SeqCst)?,
            "atomic_store_relaxed" => this.atomic_store(args, AtomicWriteOp::Relaxed)?,
            "atomic_store_rel" => this.atomic_store(args, AtomicWriteOp::Release)?,

            "atomic_fence_acq" => this.atomic_fence(args, AtomicFenceOp::Acquire)?,
            "atomic_fence_rel" => this.atomic_fence(args, AtomicFenceOp::Release)?,
            "atomic_fence_acqrel" => this.atomic_fence(args, AtomicFenceOp::AcqRel)?,
            "atomic_fence" => this.atomic_fence(args, AtomicFenceOp::SeqCst)?,

            "atomic_singlethreadfence_acq" => this.compiler_fence(args, AtomicFenceOp::Acquire)?,
            "atomic_singlethreadfence_rel" => this.compiler_fence(args, AtomicFenceOp::Release)?,
            "atomic_singlethreadfence_acqrel" =>
                this.compiler_fence(args, AtomicFenceOp::AcqRel)?,
            "atomic_singlethreadfence" => this.compiler_fence(args, AtomicFenceOp::SeqCst)?,

            "atomic_xchg" => this.atomic_exchange(args, dest, AtomicRwOp::SeqCst)?,
            "atomic_xchg_acq" => this.atomic_exchange(args, dest, AtomicRwOp::Acquire)?,
            "atomic_xchg_rel" => this.atomic_exchange(args, dest, AtomicRwOp::Release)?,
            "atomic_xchg_acqrel" => this.atomic_exchange(args, dest, AtomicRwOp::AcqRel)?,
            "atomic_xchg_relaxed" => this.atomic_exchange(args, dest, AtomicRwOp::Relaxed)?,

            #[rustfmt::skip]
            "atomic_cxchg" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::SeqCst, AtomicReadOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_cxchg_acq" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::Acquire, AtomicReadOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_cxchg_rel" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::Release, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchg_acqrel" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::AcqRel, AtomicReadOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_cxchg_relaxed" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::Relaxed, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchg_acq_failrelaxed" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::Acquire, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchg_acqrel_failrelaxed" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::AcqRel, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchg_failrelaxed" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::SeqCst, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchg_failacq" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::SeqCst, AtomicReadOp::Acquire)?,

            #[rustfmt::skip]
            "atomic_cxchgweak" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::SeqCst, AtomicReadOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_acq" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::Acquire, AtomicReadOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_rel" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::Release, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_acqrel" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::AcqRel, AtomicReadOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_relaxed" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::Relaxed, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_acq_failrelaxed" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::Acquire, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_acqrel_failrelaxed" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::AcqRel, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_failrelaxed" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::SeqCst, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_failacq" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::SeqCst, AtomicReadOp::Acquire)?,

            #[rustfmt::skip]
            "atomic_or" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_or_acq" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_or_rel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOp::Release)?,
            #[rustfmt::skip]
            "atomic_or_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOp::AcqRel)?,
            #[rustfmt::skip]
            "atomic_or_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_xor" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_xor_acq" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_xor_rel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOp::Release)?,
            #[rustfmt::skip]
            "atomic_xor_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOp::AcqRel)?,
            #[rustfmt::skip]
            "atomic_xor_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_and" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_and_acq" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_and_rel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOp::Release)?,
            #[rustfmt::skip]
            "atomic_and_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOp::AcqRel)?,
            #[rustfmt::skip]
            "atomic_and_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_nand" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_nand_acq" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_nand_rel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOp::Release)?,
            #[rustfmt::skip]
            "atomic_nand_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOp::AcqRel)?,
            #[rustfmt::skip]
            "atomic_nand_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_xadd" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_xadd_acq" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_xadd_rel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOp::Release)?,
            #[rustfmt::skip]
            "atomic_xadd_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOp::AcqRel)?,
            #[rustfmt::skip]
            "atomic_xadd_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_xsub" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_xsub_acq" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_xsub_rel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOp::Release)?,
            #[rustfmt::skip]
            "atomic_xsub_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOp::AcqRel)?,
            #[rustfmt::skip]
            "atomic_xsub_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOp::Relaxed)?,
            "atomic_min" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::SeqCst)?,
            "atomic_min_acq" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::Acquire)?,
            "atomic_min_rel" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::Release)?,
            "atomic_min_acqrel" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::AcqRel)?,
            "atomic_min_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::Relaxed)?,
            "atomic_max" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::SeqCst)?,
            "atomic_max_acq" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::Acquire)?,
            "atomic_max_rel" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::Release)?,
            "atomic_max_acqrel" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::AcqRel)?,
            "atomic_max_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::Relaxed)?,
            "atomic_umin" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::SeqCst)?,
            "atomic_umin_acq" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::Acquire)?,
            "atomic_umin_rel" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::Release)?,
            "atomic_umin_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::AcqRel)?,
            "atomic_umin_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::Relaxed)?,
            "atomic_umax" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::SeqCst)?,
            "atomic_umax_acq" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::Acquire)?,
            "atomic_umax_rel" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::Release)?,
            "atomic_umax_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::AcqRel)?,
            "atomic_umax_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::Relaxed)?,

            // Query type information
            "assert_zero_valid" | "assert_uninit_valid" => {
                let &[] = check_arg_count(args)?;
                let ty = instance.substs.type_at(0);
                let layout = this.layout_of(ty)?;
                // Abort here because the caller might not be panic safe.
                if layout.abi.is_uninhabited() {
                    // Use this message even for the other intrinsics, as that's what codegen does
                    throw_machine_stop!(TerminationInfo::Abort(format!(
                        "aborted execution: attempted to instantiate uninhabited type `{}`",
                        ty
                    )))
                }
                if intrinsic_name == "assert_zero_valid"
                    && !layout.might_permit_raw_init(this, /*zero:*/ true)
                {
                    throw_machine_stop!(TerminationInfo::Abort(format!(
                        "aborted execution: attempted to zero-initialize type `{}`, which is invalid",
                        ty
                    )))
                }
                if intrinsic_name == "assert_uninit_valid"
                    && !layout.might_permit_raw_init(this, /*zero:*/ false)
                {
                    throw_machine_stop!(TerminationInfo::Abort(format!(
                        "aborted execution: attempted to leave type `{}` uninitialized, which is invalid",
                        ty
                    )))
                }
            }

            // Other
            "exact_div" => {
                let &[ref num, ref denom] = check_arg_count(args)?;
                this.exact_div(&this.read_immediate(num)?, &this.read_immediate(denom)?, dest)?;
            }

            "try" => return this.handle_try(args, dest, ret),

            "breakpoint" => {
                let &[] = check_arg_count(args)?;
                // normally this would raise a SIGTRAP, which aborts if no debugger is connected
                throw_machine_stop!(TerminationInfo::Abort("Trace/breakpoint trap".to_string()))
            }

            name => throw_unsup_format!("unimplemented intrinsic: {}", name),
        }

        trace!("{:?}", this.dump_place(**dest));
        this.go_to_block(ret);
        Ok(())
    }

    fn atomic_load(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        atomic: AtomicReadOp,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let &[ref place] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;

        // make sure it fits into a scalar; otherwise it cannot be atomic
        let val = this.read_scalar_atomic(&place, atomic)?;

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.memory.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;
        // Perform regular access.
        this.write_scalar(val, dest)?;
        Ok(())
    }

    fn atomic_store(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        atomic: AtomicWriteOp,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let &[ref place, ref val] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;
        let val = this.read_scalar(val)?; // make sure it fits into a scalar; otherwise it cannot be atomic

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.memory.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;

        // Perform atomic store
        this.write_scalar_atomic(val, &place, atomic)?;
        Ok(())
    }

    fn compiler_fence(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        atomic: AtomicFenceOp,
    ) -> InterpResult<'tcx> {
        let &[] = check_arg_count(args)?;
        let _ = atomic;
        //FIXME: compiler fences are currently ignored
        Ok(())
    }

    fn atomic_fence(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        atomic: AtomicFenceOp,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let &[] = check_arg_count(args)?;
        this.validate_atomic_fence(atomic)?;
        Ok(())
    }

    fn atomic_op(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        atomic_op: AtomicOp,
        atomic: AtomicRwOp,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let &[ref place, ref rhs] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;

        if !place.layout.ty.is_integral() {
            bug!("Atomic arithmetic operations only work on integer types");
        }
        let rhs = this.read_immediate(rhs)?;

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.memory.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;

        match atomic_op {
            AtomicOp::Min => {
                let old = this.atomic_min_max_scalar(&place, rhs, true, atomic)?;
                this.write_immediate(*old, &dest)?; // old value is returned
                Ok(())
            }
            AtomicOp::Max => {
                let old = this.atomic_min_max_scalar(&place, rhs, false, atomic)?;
                this.write_immediate(*old, &dest)?; // old value is returned
                Ok(())
            }
            AtomicOp::MirOp(op, neg) => {
                let old = this.atomic_op_immediate(&place, &rhs, op, neg, atomic)?;
                this.write_immediate(*old, dest)?; // old value is returned
                Ok(())
            }
        }
    }

    fn atomic_exchange(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        atomic: AtomicRwOp,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let &[ref place, ref new] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;
        let new = this.read_scalar(new)?;

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.memory.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;

        let old = this.atomic_exchange_scalar(&place, new, atomic)?;
        this.write_scalar(old, dest)?; // old value is returned
        Ok(())
    }

    fn atomic_compare_exchange_impl(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        success: AtomicRwOp,
        fail: AtomicReadOp,
        can_fail_spuriously: bool,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let &[ref place, ref expect_old, ref new] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;
        let expect_old = this.read_immediate(expect_old)?; // read as immediate for the sake of `binary_op()`
        let new = this.read_scalar(new)?;

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.memory.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;

        let old = this.atomic_compare_exchange_scalar(
            &place,
            &expect_old,
            new,
            success,
            fail,
            can_fail_spuriously,
        )?;

        // Return old value.
        this.write_immediate(old, dest)?;
        Ok(())
    }

    fn atomic_compare_exchange(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        success: AtomicRwOp,
        fail: AtomicReadOp,
    ) -> InterpResult<'tcx> {
        self.atomic_compare_exchange_impl(args, dest, success, fail, false)
    }

    fn atomic_compare_exchange_weak(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        success: AtomicRwOp,
        fail: AtomicReadOp,
    ) -> InterpResult<'tcx> {
        self.atomic_compare_exchange_impl(args, dest, success, fail, true)
    }

    fn float_to_int_unchecked<F>(
        &self,
        f: F,
        dest_ty: ty::Ty<'tcx>,
    ) -> InterpResult<'tcx, Scalar<Tag>>
    where
        F: Float + Into<Scalar<Tag>>,
    {
        let this = self.eval_context_ref();

        // Step 1: cut off the fractional part of `f`. The result of this is
        // guaranteed to be precisely representable in IEEE floats.
        let f = f.round_to_integral(Round::TowardZero).value;

        // Step 2: Cast the truncated float to the target integer type and see if we lose any information in this step.
        Ok(match dest_ty.kind() {
            // Unsigned
            ty::Uint(t) => {
                let size = Integer::from_uint_ty(this, *t).size();
                let res = f.to_u128(size.bits_usize());
                if res.status.is_empty() {
                    // No status flags means there was no further rounding or other loss of precision.
                    Scalar::from_uint(res.value, size)
                } else {
                    // `f` was not representable in this integer type.
                    throw_ub_format!(
                        "`float_to_int_unchecked` intrinsic called on {} which cannot be represented in target type `{:?}`",
                        f,
                        dest_ty,
                    );
                }
            }
            // Signed
            ty::Int(t) => {
                let size = Integer::from_int_ty(this, *t).size();
                let res = f.to_i128(size.bits_usize());
                if res.status.is_empty() {
                    // No status flags means there was no further rounding or other loss of precision.
                    Scalar::from_int(res.value, size)
                } else {
                    // `f` was not representable in this integer type.
                    throw_ub_format!(
                        "`float_to_int_unchecked` intrinsic called on {} which cannot be represented in target type `{:?}`",
                        f,
                        dest_ty,
                    );
                }
            }
            // Nothing else
            _ => bug!("`float_to_int_unchecked` called with non-int output type {:?}", dest_ty),
        })
    }
}
