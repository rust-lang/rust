use std::iter;

use log::trace;

use rustc_attr as attr;
use rustc_ast::ast::FloatTy;
use rustc_middle::{mir, ty};
use rustc_middle::ty::layout::IntegerExt;
use rustc_apfloat::{Float, Round};
use rustc_target::abi::{Align, Integer, LayoutOf};
use rustc_span::symbol::sym;

use crate::*;
use helpers::check_arg_count;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn call_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
        _unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let intrinsic_name = this.tcx.item_name(instance.def_id());
        // We want to overwrite some of the intrinsic implementations that CTFE uses.
        let prefer_miri_intrinsic = match intrinsic_name {
            sym::ptr_guaranteed_eq | sym::ptr_guaranteed_ne => true,
            _ => false,
        };

        if !prefer_miri_intrinsic && this.emulate_intrinsic(instance, args, ret)? {
            return Ok(());
        }

        // All supported intrinsics have a return place.
        let intrinsic_name = &*intrinsic_name.as_str();
        let (dest, ret) = match ret {
            None => throw_unsup_format!("unimplemented (diverging) intrinsic: {}", intrinsic_name),
            Some(p) => p,
        };

        // Then handle terminating intrinsics.
        match intrinsic_name {
            // Miri overwriting CTFE intrinsics.
            "ptr_guaranteed_eq" => {
                let &[left, right] = check_arg_count(args)?;
                let left = this.read_immediate(left)?;
                let right = this.read_immediate(right)?;
                this.binop_ignore_overflow(mir::BinOp::Eq, left, right, dest)?;
            }
            "ptr_guaranteed_ne" => {
                let &[left, right] = check_arg_count(args)?;
                let left = this.read_immediate(left)?;
                let right = this.read_immediate(right)?;
                this.binop_ignore_overflow(mir::BinOp::Ne, left, right, dest)?;
            }

            // Raw memory accesses
            #[rustfmt::skip]
            | "copy"
            | "copy_nonoverlapping"
            => {
                let &[src, dest, count] = check_arg_count(args)?;
                let elem_ty = instance.substs.type_at(0);
                let elem_layout = this.layout_of(elem_ty)?;
                let count = this.read_scalar(count)?.to_machine_usize(this)?;
                let elem_align = elem_layout.align.abi;

                let size = elem_layout.size.checked_mul(count, this)
                    .ok_or_else(|| err_ub_format!("overflow computing total size of `{}`", intrinsic_name))?;
                let src = this.read_scalar(src)?.check_init()?;
                let src = this.memory.check_ptr_access(src, size, elem_align)?;
                let dest = this.read_scalar(dest)?.check_init()?;
                let dest = this.memory.check_ptr_access(dest, size, elem_align)?;

                if let (Some(src), Some(dest)) = (src, dest) {
                    this.memory.copy(
                        src,
                        dest,
                        size,
                        intrinsic_name.ends_with("_nonoverlapping"),
                    )?;
                }
            }

            "move_val_init" => {
                let &[place, dest] = check_arg_count(args)?;
                let place = this.deref_operand(place)?;
                this.copy_op(dest, place.into())?;
            }

            "volatile_load" => {
                let &[place] = check_arg_count(args)?;
                let place = this.deref_operand(place)?;
                this.copy_op(place.into(), dest)?;
            }
            "volatile_store" => {
                let &[place, dest] = check_arg_count(args)?;
                let place = this.deref_operand(place)?;
                this.copy_op(dest, place.into())?;
            }

            "write_bytes" => {
                let &[ptr, val_byte, count] = check_arg_count(args)?;
                let ty = instance.substs.type_at(0);
                let ty_layout = this.layout_of(ty)?;
                let val_byte = this.read_scalar(val_byte)?.to_u8()?;
                let ptr = this.read_scalar(ptr)?.check_init()?;
                let count = this.read_scalar(count)?.to_machine_usize(this)?;
                let byte_count = ty_layout.size.checked_mul(count, this)
                    .ok_or_else(|| err_ub_format!("overflow computing total size of `write_bytes`"))?;
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
                let &[f] = check_arg_count(args)?;
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
                let &[f] = check_arg_count(args)?;
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
                let &[a, b] = check_arg_count(args)?;
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
                this.binop_ignore_overflow(op, a, b, dest)?;
            }

            #[rustfmt::skip]
            | "minnumf32"
            | "maxnumf32"
            | "copysignf32"
            => {
                let &[a, b] = check_arg_count(args)?;
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
                let &[a, b] = check_arg_count(args)?;
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
                let &[f, f2] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(f)?.to_u32()?);
                let f2 = f32::from_bits(this.read_scalar(f2)?.to_u32()?);
                this.write_scalar(Scalar::from_u32(f.powf(f2).to_bits()), dest)?;
            }

            "powf64" => {
                let &[f, f2] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(f)?.to_u64()?);
                let f2 = f64::from_bits(this.read_scalar(f2)?.to_u64()?);
                this.write_scalar(Scalar::from_u64(f.powf(f2).to_bits()), dest)?;
            }

            "fmaf32" => {
                let &[a, b, c] = check_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f32()?;
                let b = this.read_scalar(b)?.to_f32()?;
                let c = this.read_scalar(c)?.to_f32()?;
                let res = a.mul_add(b, c).value;
                this.write_scalar(Scalar::from_f32(res), dest)?;
            }

            "fmaf64" => {
                let &[a, b, c] = check_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f64()?;
                let b = this.read_scalar(b)?.to_f64()?;
                let c = this.read_scalar(c)?.to_f64()?;
                let res = a.mul_add(b, c).value;
                this.write_scalar(Scalar::from_f64(res), dest)?;
            }

            "powif32" => {
                let &[f, i] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(f)?.to_u32()?);
                let i = this.read_scalar(i)?.to_i32()?;
                this.write_scalar(Scalar::from_u32(f.powi(i).to_bits()), dest)?;
            }

            "powif64" => {
                let &[f, i] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(f)?.to_u64()?);
                let i = this.read_scalar(i)?.to_i32()?;
                this.write_scalar(Scalar::from_u64(f.powi(i).to_bits()), dest)?;
            }

            "float_to_int_unchecked" => {
                let &[val] = check_arg_count(args)?;
                let val = this.read_immediate(val)?;

                let res = match val.layout.ty.kind {
                    ty::Float(FloatTy::F32) => {
                        this.float_to_int_unchecked(val.to_scalar()?.to_f32()?, dest.layout.ty)?
                    }
                    ty::Float(FloatTy::F64) => {
                        this.float_to_int_unchecked(val.to_scalar()?.to_f64()?, dest.layout.ty)?
                    }
                    _ => bug!("`float_to_int_unchecked` called with non-float input type {:?}", val.layout.ty),
                };

                this.write_scalar(res, dest)?;
            }

            // Atomic operations
            #[rustfmt::skip]
            | "atomic_load"
            | "atomic_load_relaxed"
            | "atomic_load_acq"
            => {
                let &[place] = check_arg_count(args)?;
                let place = this.deref_operand(place)?;
                let val = this.read_scalar(place.into())?; // make sure it fits into a scalar; otherwise it cannot be atomic

                // Check alignment requirements. Atomics must always be aligned to their size,
                // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
                // be 8-aligned).
                let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
                this.memory.check_ptr_access(place.ptr, place.layout.size, align)?;

                this.write_scalar(val, dest)?;
            }

            #[rustfmt::skip]
            | "atomic_store"
            | "atomic_store_relaxed"
            | "atomic_store_rel"
            => {
                let &[place, val] = check_arg_count(args)?;
                let place = this.deref_operand(place)?;
                let val = this.read_scalar(val)?; // make sure it fits into a scalar; otherwise it cannot be atomic

                // Check alignment requirements. Atomics must always be aligned to their size,
                // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
                // be 8-aligned).
                let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
                this.memory.check_ptr_access(place.ptr, place.layout.size, align)?;

                this.write_scalar(val, place.into())?;
            }

            #[rustfmt::skip]
            | "atomic_fence_acq"
            | "atomic_fence_rel"
            | "atomic_fence_acqrel"
            | "atomic_fence"
            | "atomic_singlethreadfence_acq"
            | "atomic_singlethreadfence_rel"
            | "atomic_singlethreadfence_acqrel"
            | "atomic_singlethreadfence"
            => {
                let &[] = check_arg_count(args)?;
                // FIXME: this will become relevant once we try to detect data races.
            }

            _ if intrinsic_name.starts_with("atomic_xchg") => {
                let &[place, new] = check_arg_count(args)?;
                let place = this.deref_operand(place)?;
                let new = this.read_scalar(new)?;
                let old = this.read_scalar(place.into())?;

                // Check alignment requirements. Atomics must always be aligned to their size,
                // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
                // be 8-aligned).
                let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
                this.memory.check_ptr_access(place.ptr, place.layout.size, align)?;

                this.write_scalar(old, dest)?; // old value is returned
                this.write_scalar(new, place.into())?;
            }

            _ if intrinsic_name.starts_with("atomic_cxchg") => {
                let &[place, expect_old, new] = check_arg_count(args)?;
                let place = this.deref_operand(place)?;
                let expect_old = this.read_immediate(expect_old)?; // read as immediate for the sake of `binary_op()`
                let new = this.read_scalar(new)?;
                let old = this.read_immediate(place.into())?; // read as immediate for the sake of `binary_op()`

                // Check alignment requirements. Atomics must always be aligned to their size,
                // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
                // be 8-aligned).
                let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
                this.memory.check_ptr_access(place.ptr, place.layout.size, align)?;

                // `binary_op` will bail if either of them is not a scalar.
                let eq = this.overflowing_binary_op(mir::BinOp::Eq, old, expect_old)?.0;
                let res = Immediate::ScalarPair(old.to_scalar_or_uninit(), eq.into());
                // Return old value.
                this.write_immediate(res, dest)?;
                // Update ptr depending on comparison.
                if eq.to_bool()? {
                    this.write_scalar(new, place.into())?;
                }
            }

            #[rustfmt::skip]
            | "atomic_or"
            | "atomic_or_acq"
            | "atomic_or_rel"
            | "atomic_or_acqrel"
            | "atomic_or_relaxed"
            | "atomic_xor"
            | "atomic_xor_acq"
            | "atomic_xor_rel"
            | "atomic_xor_acqrel"
            | "atomic_xor_relaxed"
            | "atomic_and"
            | "atomic_and_acq"
            | "atomic_and_rel"
            | "atomic_and_acqrel"
            | "atomic_and_relaxed"
            | "atomic_nand"
            | "atomic_nand_acq"
            | "atomic_nand_rel"
            | "atomic_nand_acqrel"
            | "atomic_nand_relaxed"
            | "atomic_xadd"
            | "atomic_xadd_acq"
            | "atomic_xadd_rel"
            | "atomic_xadd_acqrel"
            | "atomic_xadd_relaxed"
            | "atomic_xsub"
            | "atomic_xsub_acq"
            | "atomic_xsub_rel"
            | "atomic_xsub_acqrel"
            | "atomic_xsub_relaxed"
            => {
                let &[place, rhs] = check_arg_count(args)?;
                let place = this.deref_operand(place)?;
                if !place.layout.ty.is_integral() {
                    bug!("Atomic arithmetic operations only work on integer types");
                }
                let rhs = this.read_immediate(rhs)?;
                let old = this.read_immediate(place.into())?;

                // Check alignment requirements. Atomics must always be aligned to their size,
                // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
                // be 8-aligned).
                let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
                this.memory.check_ptr_access(place.ptr, place.layout.size, align)?;

                this.write_immediate(*old, dest)?; // old value is returned
                let (op, neg) = match intrinsic_name.split('_').nth(1).unwrap() {
                    "or" => (mir::BinOp::BitOr, false),
                    "xor" => (mir::BinOp::BitXor, false),
                    "and" => (mir::BinOp::BitAnd, false),
                    "xadd" => (mir::BinOp::Add, false),
                    "xsub" => (mir::BinOp::Sub, false),
                    "nand" => (mir::BinOp::BitAnd, true),
                    _ => bug!(),
                };
                // Atomics wrap around on overflow.
                let val = this.binary_op(op, old, rhs)?;
                let val = if neg { this.unary_op(mir::UnOp::Not, val)? } else { val };
                this.write_immediate(*val, place.into())?;
            }

            // Query type information
            "assert_inhabited" |
            "assert_zero_valid" |
            "assert_uninit_valid" => {
                let &[] = check_arg_count(args)?;
                let ty = instance.substs.type_at(0);
                let layout = this.layout_of(ty)?;
                // Abort here because the caller might not be panic safe.
                if layout.abi.is_uninhabited() {
                    throw_machine_stop!(TerminationInfo::Abort(Some(format!("attempted to instantiate uninhabited type `{}`", ty))))
                }
                if intrinsic_name == "assert_zero_valid" && !layout.might_permit_raw_init(this, /*zero:*/ true).unwrap() {
                    throw_machine_stop!(TerminationInfo::Abort(Some(format!("attempted to zero-initialize type `{}`, which is invalid", ty))))
                }
                if intrinsic_name == "assert_uninit_valid" && !layout.might_permit_raw_init(this, /*zero:*/ false).unwrap() {
                    throw_machine_stop!(TerminationInfo::Abort(Some(format!("attempted to leave type `{}` uninitialized, which is invalid", ty))))
                }
            }

            // Other
            "assume" => {
                let &[cond] = check_arg_count(args)?;
                let cond = this.read_scalar(cond)?.check_init()?.to_bool()?;
                if !cond {
                    throw_ub_format!("`assume` intrinsic called with `false`");
                }
            }

            "exact_div" => {
                let &[num, denom] = check_arg_count(args)?;
                this.exact_div(this.read_immediate(num)?, this.read_immediate(denom)?, dest)?;
            }

            "forget" => {
                // We get an argument... and forget about it.
                let &[_] = check_arg_count(args)?;
            }

            "try" => return this.handle_try(args, dest, ret),

            name => throw_unsup_format!("unimplemented intrinsic: {}", name),
        }

        trace!("{:?}", this.dump_place(*dest));
        this.go_to_block(ret);
        Ok(())
    }

    fn float_to_int_unchecked<F>(
        &self,
        f: F,
        dest_ty: ty::Ty<'tcx>,
    ) -> InterpResult<'tcx, Scalar<Tag>>
    where
        F: Float + Into<Scalar<Tag>>
    {
        let this = self.eval_context_ref();

        // Step 1: cut off the fractional part of `f`. The result of this is
        // guaranteed to be precisely representable in IEEE floats.
        let f = f.round_to_integral(Round::TowardZero).value;

        // Step 2: Cast the truncated float to the target integer type and see if we lose any information in this step.
        Ok(match dest_ty.kind {
            // Unsigned
            ty::Uint(t) => {
                let size = Integer::from_attr(this, attr::IntType::UnsignedInt(t)).size();
                let res = f.to_u128(size.bits_usize());
                if res.status.is_empty() {
                    // No status flags means there was no further rounding or other loss of precision.
                    Scalar::from_uint(res.value, size)
                } else {
                    // `f` was not representable in this integer type.
                    throw_ub_format!(
                        "`float_to_int_unchecked` intrinsic called on {} which cannot be represented in target type `{:?}`",
                        f, dest_ty,
                    );
                }
            }
            // Signed
            ty::Int(t) => {
                let size = Integer::from_attr(this, attr::IntType::SignedInt(t)).size();
                let res = f.to_i128(size.bits_usize());
                if res.status.is_empty() {
                    // No status flags means there was no further rounding or other loss of precision.
                    Scalar::from_int(res.value, size)
                } else {
                    // `f` was not representable in this integer type.
                    throw_ub_format!(
                        "`float_to_int_unchecked` intrinsic called on {} which cannot be represented in target type `{:?}`",
                        f, dest_ty,
                    );
                }
            }
            // Nothing else
            _ => bug!("`float_to_int_unchecked` called with non-int output type {:?}", dest_ty),
        })
    }
}
