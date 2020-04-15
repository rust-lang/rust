use std::iter;
use std::convert::TryFrom;

use rustc_middle::{mir, ty};
use rustc_apfloat::Float;
use rustc_target::abi::{Align, LayoutOf};

use crate::*;

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn call_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
        unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        if this.emulate_intrinsic(instance, args, ret)? {
            return Ok(());
        }
        let substs = instance.substs;

        // All these intrinsics take raw pointers, so if we access memory directly
        // (as opposed to through a place), we have to remember to erase any tag
        // that might still hang around!
        let intrinsic_name = &*this.tcx.item_name(instance.def_id()).as_str();

        // First handle intrinsics without return place.
        let (dest, ret) = match ret {
            None => match intrinsic_name {
                "miri_start_panic" => return this.handle_miri_start_panic(args, unwind),
                "unreachable" => throw_ub!(Unreachable),
                _ => throw_unsup_format!("unimplemented (diverging) intrinsic: {}", intrinsic_name),
            },
            Some(p) => p,
        };

        // Then handle terminating intrinsics.
        match intrinsic_name {
            // Raw memory accesses
            #[rustfmt::skip]
            | "copy"
            | "copy_nonoverlapping"
            => {
                let elem_ty = substs.type_at(0);
                let elem_layout = this.layout_of(elem_ty)?;
                let count = this.read_scalar(args[2])?.to_machine_usize(this)?;
                let elem_align = elem_layout.align.abi;

                let size = elem_layout.size.checked_mul(count, this)
                    .ok_or_else(|| err_ub_format!("overflow computing total size of `{}`", intrinsic_name))?;
                let src = this.read_scalar(args[0])?.not_undef()?;
                let src = this.memory.check_ptr_access(src, size, elem_align)?;
                let dest = this.read_scalar(args[1])?.not_undef()?;
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
                let place = this.deref_operand(args[0])?;
                this.copy_op(args[1], place.into())?;
            }

            "volatile_load" => {
                let place = this.deref_operand(args[0])?;
                this.copy_op(place.into(), dest)?;
            }
            "volatile_store" => {
                let place = this.deref_operand(args[0])?;
                this.copy_op(args[1], place.into())?;
            }

            "write_bytes" => {
                let ty = substs.type_at(0);
                let ty_layout = this.layout_of(ty)?;
                let val_byte = this.read_scalar(args[1])?.to_u8()?;
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let count = this.read_scalar(args[2])?.to_machine_usize(this)?;
                let byte_count = ty_layout.size.checked_mul(count, this)
                    .ok_or_else(|| err_ub_format!("overflow computing total size of `write_bytes`"))?;
                this.memory
                    .write_bytes(ptr, iter::repeat(val_byte).take(byte_count.bytes() as usize))?;
            }

            // Pointer arithmetic
            "arith_offset" => {
                let offset = this.read_scalar(args[1])?.to_machine_isize(this)?;
                let ptr = this.read_scalar(args[0])?.not_undef()?;

                let pointee_ty = substs.type_at(0);
                let pointee_size = i64::try_from(this.layout_of(pointee_ty)?.size.bytes()).unwrap();
                let offset = offset.overflowing_mul(pointee_size).0;
                let result_ptr = ptr.ptr_wrapping_signed_offset(offset, this);
                this.write_scalar(result_ptr, dest)?;
            }
            "offset" => {
                let offset = this.read_scalar(args[1])?.to_machine_isize(this)?;
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let result_ptr = this.pointer_offset_inbounds(ptr, substs.type_at(0), offset)?;
                this.write_scalar(result_ptr, dest)?;
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
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(args[0])?.to_u32()?);
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
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(args[0])?.to_u64()?);
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
                let a = this.read_immediate(args[0])?;
                let b = this.read_immediate(args[1])?;
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
                let a = this.read_scalar(args[0])?.to_f32()?;
                let b = this.read_scalar(args[1])?.to_f32()?;
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
                let a = this.read_scalar(args[0])?.to_f64()?;
                let b = this.read_scalar(args[1])?.to_f64()?;
                let res = match intrinsic_name {
                    "minnumf64" => a.min(b),
                    "maxnumf64" => a.max(b),
                    "copysignf64" => a.copy_sign(b),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_f64(res), dest)?;
            }
            
            "powf32" => {
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(args[0])?.to_u32()?);
                let f2 = f32::from_bits(this.read_scalar(args[1])?.to_u32()?);
                this.write_scalar(Scalar::from_u32(f.powf(f2).to_bits()), dest)?;
            }

            "powf64" => {
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(args[0])?.to_u64()?);
                let f2 = f64::from_bits(this.read_scalar(args[1])?.to_u64()?);
                this.write_scalar(Scalar::from_u64(f.powf(f2).to_bits()), dest)?;
            }

            "fmaf32" => {
                let a = this.read_scalar(args[0])?.to_f32()?;
                let b = this.read_scalar(args[1])?.to_f32()?;
                let c = this.read_scalar(args[2])?.to_f32()?;
                let res = a.mul_add(b, c).value;
                this.write_scalar(Scalar::from_f32(res), dest)?;
            }

            "fmaf64" => {
                let a = this.read_scalar(args[0])?.to_f64()?;
                let b = this.read_scalar(args[1])?.to_f64()?;
                let c = this.read_scalar(args[2])?.to_f64()?;
                let res = a.mul_add(b, c).value;
                this.write_scalar(Scalar::from_f64(res), dest)?;
            }

            "powif32" => {
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(args[0])?.to_u32()?);
                let i = this.read_scalar(args[1])?.to_i32()?;
                this.write_scalar(Scalar::from_u32(f.powi(i).to_bits()), dest)?;
            }

            "powif64" => {
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(args[0])?.to_u64()?);
                let i = this.read_scalar(args[1])?.to_i32()?;
                this.write_scalar(Scalar::from_u64(f.powi(i).to_bits()), dest)?;
            }

            // Atomic operations
            #[rustfmt::skip]
            | "atomic_load"
            | "atomic_load_relaxed"
            | "atomic_load_acq"
            => {
                let place = this.deref_operand(args[0])?;
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
                let place = this.deref_operand(args[0])?;
                let val = this.read_scalar(args[1])?; // make sure it fits into a scalar; otherwise it cannot be atomic

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
                // we are inherently singlethreaded and singlecored, this is a nop
            }

            _ if intrinsic_name.starts_with("atomic_xchg") => {
                let place = this.deref_operand(args[0])?;
                let new = this.read_scalar(args[1])?;
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
                let place = this.deref_operand(args[0])?;
                let expect_old = this.read_immediate(args[1])?; // read as immediate for the sake of `binary_op()`
                let new = this.read_scalar(args[2])?;
                let old = this.read_immediate(place.into())?; // read as immediate for the sake of `binary_op()`

                // Check alignment requirements. Atomics must always be aligned to their size,
                // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
                // be 8-aligned).
                let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
                this.memory.check_ptr_access(place.ptr, place.layout.size, align)?;

                // `binary_op` will bail if either of them is not a scalar.
                let eq = this.overflowing_binary_op(mir::BinOp::Eq, old, expect_old)?.0;
                let res = Immediate::ScalarPair(old.to_scalar_or_undef(), eq.into());
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
                let place = this.deref_operand(args[0])?;
                if !place.layout.ty.is_integral() {
                    bug!("Atomic arithmetic operations only work on integer types");
                }
                let rhs = this.read_immediate(args[1])?;
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
                let ty = substs.type_at(0);
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

            "min_align_of_val" => {
                let mplace = this.deref_operand(args[0])?;
                let (_, align) = this
                    .size_and_align_of_mplace(mplace)?
                    .expect("size_of_val called on extern type");
                this.write_scalar(Scalar::from_machine_usize(align.bytes(), this), dest)?;
            }

            "size_of_val" => {
                let mplace = this.deref_operand(args[0])?;
                let (size, _) = this
                    .size_and_align_of_mplace(mplace)?
                    .expect("size_of_val called on extern type");
                this.write_scalar(Scalar::from_machine_usize(size.bytes(), this), dest)?;
            }

            // Other
            "assume" => {
                let cond = this.read_scalar(args[0])?.to_bool()?;
                if !cond {
                    throw_ub_format!("`assume` intrinsic called with `false`");
                }
            }

            "exact_div" =>
                this.exact_div(this.read_immediate(args[0])?, this.read_immediate(args[1])?, dest)?,

            "forget" => {
                // We get an argument... and forget about it.
            }

            #[rustfmt::skip]
            | "likely"
            | "unlikely"
            => {
                // These just return their argument
                let b = this.read_immediate(args[0])?;
                this.write_immediate(*b, dest)?;
            }

            "try" => return this.handle_try(args, dest, ret),

            name => throw_unsup_format!("unimplemented intrinsic: {}", name),
        }

        this.dump_place(*dest);
        this.go_to_block(ret);
        Ok(())
    }
}
