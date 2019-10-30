use std::iter;

use rustc_apfloat::Float;
use rustc::mir;
use rustc::mir::interpret::{InterpResult, PointerArithmetic};
use rustc::ty::layout::{self, LayoutOf, Size, Align};
use rustc::ty;
use syntax::source_map::Span;

use crate::{
    PlaceTy, OpTy, Immediate, Scalar, Tag,
    OperatorEvalContextExt
};

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn call_intrinsic(
        &mut self,
        span: Span,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        dest: PlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        if this.emulate_intrinsic(span, instance, args, dest)? {
            return Ok(());
        }
        let tcx = &{this.tcx.tcx};
        let substs = instance.substs;

        // All these intrinsics take raw pointers, so if we access memory directly
        // (as opposed to through a place), we have to remember to erase any tag
        // that might still hang around!

        let intrinsic_name = &*tcx.item_name(instance.def_id()).as_str();
        match intrinsic_name {
            "arith_offset" => {
                let offset = this.read_scalar(args[1])?.to_isize(this)?;
                let ptr = this.read_scalar(args[0])?.not_undef()?;

                let pointee_ty = substs.type_at(0);
                let pointee_size = this.layout_of(pointee_ty)?.size.bytes() as i64;
                let offset = offset.overflowing_mul(pointee_size).0;
                let result_ptr = ptr.ptr_wrapping_signed_offset(offset, this);
                this.write_scalar(result_ptr, dest)?;
            }

            "assume" => {
                let cond = this.read_scalar(args[0])?.to_bool()?;
                if !cond {
                    throw_ub_format!("`assume` intrinsic called with `false`");
                }
            }

            "volatile_load" => {
                let place = this.deref_operand(args[0])?;
                this.copy_op(place.into(), dest)?;
            }

            "volatile_store" => {
                let place = this.deref_operand(args[0])?;
                this.copy_op(args[1], place.into())?;
            }

            "atomic_load" |
            "atomic_load_relaxed" |
            "atomic_load_acq" => {
                let place = this.deref_operand(args[0])?;
                let val = this.read_scalar(place.into())?; // make sure it fits into a scalar; otherwise it cannot be atomic

                // Check alignment requirements. Atomics must always be aligned to their size,
                // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
                // be 8-aligned).
                let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
                this.memory.check_ptr_access(place.ptr, place.layout.size, align)?;

                this.write_scalar(val, dest)?;
            }

            "atomic_store" |
            "atomic_store_relaxed" |
            "atomic_store_rel" => {
                let place = this.deref_operand(args[0])?;
                let val = this.read_scalar(args[1])?; // make sure it fits into a scalar; otherwise it cannot be atomic

                // Check alignment requirements. Atomics must always be aligned to their size,
                // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
                // be 8-aligned).
                let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
                this.memory.check_ptr_access(place.ptr, place.layout.size, align)?;

                this.write_scalar(val, place.into())?;
            }

            "atomic_fence_acq" |
            "atomic_fence_rel" |
            "atomic_fence_acqrel" |
            "atomic_fence" => {
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

                // binary_op will bail if either of them is not a scalar
                let eq = this.overflowing_binary_op(mir::BinOp::Eq, old, expect_old)?.0;
                let res = Immediate::ScalarPair(old.to_scalar_or_undef(), eq.into());
                this.write_immediate(res, dest)?; // old value is returned
                // update ptr depending on comparison
                if eq.to_bool()? {
                    this.write_scalar(new, place.into())?;
                }
            }

            "atomic_or" |
            "atomic_or_acq" |
            "atomic_or_rel" |
            "atomic_or_acqrel" |
            "atomic_or_relaxed" |
            "atomic_xor" |
            "atomic_xor_acq" |
            "atomic_xor_rel" |
            "atomic_xor_acqrel" |
            "atomic_xor_relaxed" |
            "atomic_and" |
            "atomic_and_acq" |
            "atomic_and_rel" |
            "atomic_and_acqrel" |
            "atomic_and_relaxed" |
            "atomic_nand" |
            "atomic_nand_acq" |
            "atomic_nand_rel" |
            "atomic_nand_acqrel" |
            "atomic_nand_relaxed" |
            "atomic_xadd" |
            "atomic_xadd_acq" |
            "atomic_xadd_rel" |
            "atomic_xadd_acqrel" |
            "atomic_xadd_relaxed" |
            "atomic_xsub" |
            "atomic_xsub_acq" |
            "atomic_xsub_rel" |
            "atomic_xsub_acqrel" |
            "atomic_xsub_relaxed" => {
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
                let val = if neg {
                    this.unary_op(mir::UnOp::Not, val)?
                } else {
                    val
                };
                this.write_immediate(*val, place.into())?;
            }

            "breakpoint" => unimplemented!(), // halt miri

            "copy" |
            "copy_nonoverlapping" => {
                let elem_ty = substs.type_at(0);
                let elem_layout = this.layout_of(elem_ty)?;
                let elem_size = elem_layout.size.bytes();
                let count = this.read_scalar(args[2])?.to_usize(this)?;
                let elem_align = elem_layout.align.abi;

                let size = Size::from_bytes(count * elem_size);
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

            "discriminant_value" => {
                let place = this.deref_operand(args[0])?;
                let discr_val = this.read_discriminant(place.into())?.0;
                this.write_scalar(Scalar::from_uint(discr_val, dest.layout.size), dest)?;
            }

            "sinf32" | "fabsf32" | "cosf32" | "sqrtf32" | "expf32" | "exp2f32" | "logf32" |
            "log10f32" | "log2f32" | "floorf32" | "ceilf32" | "truncf32" | "roundf32" => {
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

            "sinf64" | "fabsf64" | "cosf64" | "sqrtf64" | "expf64" | "exp2f64" | "logf64" |
            "log10f64" | "log2f64" | "floorf64" | "ceilf64" | "truncf64" | "roundf64" => {
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

            "fadd_fast" | "fsub_fast" | "fmul_fast" | "fdiv_fast" | "frem_fast" => {
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

            "minnumf32" | "maxnumf32" => {
                let a = this.read_scalar(args[0])?.to_f32()?;
                let b = this.read_scalar(args[1])?.to_f32()?;
                let res = if intrinsic_name.starts_with("min") {
                    a.min(b)
                } else {
                    a.max(b)
                };
                this.write_scalar(Scalar::from_f32(res), dest)?;
            }

            "minnumf64" | "maxnumf64" => {
                let a = this.read_scalar(args[0])?.to_f64()?;
                let b = this.read_scalar(args[1])?.to_f64()?;
                let res = if intrinsic_name.starts_with("min") {
                    a.min(b)
                } else {
                    a.max(b)
                };
                this.write_scalar(Scalar::from_f64(res), dest)?;
            }

            "exact_div" => {
                // Performs an exact division, resulting in undefined behavior where
                // `x % y != 0` or `y == 0` or `x == T::min_value() && y == -1`
                let a = this.read_immediate(args[0])?;
                let b = this.read_immediate(args[1])?;
                // check x % y != 0
                if this.overflowing_binary_op(mir::BinOp::Rem, a, b)?.0.to_bits(dest.layout.size)? != 0 {
                    // Check if `b` is -1, which is the "min_value / -1" case.
                    let minus1 = Scalar::from_int(-1, dest.layout.size);
                    return Err(if b.to_scalar().unwrap() == minus1 {
                        err_ub_format!("exact_div: result of dividing MIN by -1 cannot be represented")
                    } else {
                        err_ub_format!("exact_div: {:?} cannot be divided by {:?} without remainder", *a, *b)
                    }.into());
                }
                this.binop_ignore_overflow(mir::BinOp::Div, a, b, dest)?;
            },

            "forget" => {}

            "likely" | "unlikely" => {
                // These just return their argument
                let b = this.read_immediate(args[0])?;
                this.write_immediate(*b, dest)?;
            }

            "init" => {
                // Check fast path: we don't want to force an allocation in case the destination is a simple value,
                // but we also do not want to create a new allocation with 0s and then copy that over.
                // FIXME: We do not properly validate in case of ZSTs and when doing it in memory!
                // However, this only affects direct calls of the intrinsic; calls to the stable
                // functions wrapping them do get their validation.
                // FIXME: should we check that the destination pointer is aligned even for ZSTs?
                if !dest.layout.is_zst() {
                    match dest.layout.abi {
                        layout::Abi::Scalar(ref s) => {
                            let x = Scalar::from_int(0, s.value.size(this));
                            this.write_scalar(x, dest)?;
                        }
                        layout::Abi::ScalarPair(ref s1, ref s2) => {
                            let x = Scalar::from_int(0, s1.value.size(this));
                            let y = Scalar::from_int(0, s2.value.size(this));
                            this.write_immediate(Immediate::ScalarPair(x.into(), y.into()), dest)?;
                        }
                        _ => {
                            // Do it in memory
                            let mplace = this.force_allocation(dest)?;
                            mplace.meta.unwrap_none(); // must be sized
                            this.memory.write_bytes(mplace.ptr, iter::repeat(0u8).take(dest.layout.size.bytes() as usize))?;
                        }
                    }
                }
            }

            "pref_align_of" => {
                let ty = substs.type_at(0);
                let layout = this.layout_of(ty)?;
                let align = layout.align.pref.bytes();
                let ptr_size = this.pointer_size();
                let align_val = Scalar::from_uint(align as u128, ptr_size);
                this.write_scalar(align_val, dest)?;
            }

            "move_val_init" => {
                let place = this.deref_operand(args[0])?;
                this.copy_op(args[1], place.into())?;
            }

            "offset" => {
                let offset = this.read_scalar(args[1])?.to_isize(this)?;
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let result_ptr = this.pointer_offset_inbounds(ptr, substs.type_at(0), offset)?;
                this.write_scalar(result_ptr, dest)?;
            }

            "panic_if_uninhabited" => {
                let ty = substs.type_at(0);
                let layout = this.layout_of(ty)?;
                if layout.abi.is_uninhabited() {
                    throw_ub_format!("Trying to instantiate uninhabited type {}", ty)
                }
            }

            "powf32" => {
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(args[0])?.to_u32()?);
                let f2 = f32::from_bits(this.read_scalar(args[1])?.to_u32()?);
                this.write_scalar(
                    Scalar::from_u32(f.powf(f2).to_bits()),
                    dest,
                )?;
            }

            "powf64" => {
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(args[0])?.to_u64()?);
                let f2 = f64::from_bits(this.read_scalar(args[1])?.to_u64()?);
                this.write_scalar(
                    Scalar::from_u64(f.powf(f2).to_bits()),
                    dest,
                )?;
            }

            "fmaf32" => {
                let a = this.read_scalar(args[0])?.to_f32()?;
                let b = this.read_scalar(args[1])?.to_f32()?;
                let c = this.read_scalar(args[2])?.to_f32()?;
                let res = a.mul_add(b, c).value;
                this.write_scalar(
                    Scalar::from_f32(res),
                    dest,
                )?;
            }

            "fmaf64" => {
                let a = this.read_scalar(args[0])?.to_f64()?;
                let b = this.read_scalar(args[1])?.to_f64()?;
                let c = this.read_scalar(args[2])?.to_f64()?;
                let res = a.mul_add(b, c).value;
                this.write_scalar(
                    Scalar::from_f64(res),
                    dest,
                )?;
            }

            "powif32" => {
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(args[0])?.to_u32()?);
                let i = this.read_scalar(args[1])?.to_i32()?;
                this.write_scalar(
                    Scalar::from_u32(f.powi(i).to_bits()),
                    dest,
                )?;
            }

            "powif64" => {
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(args[0])?.to_u64()?);
                let i = this.read_scalar(args[1])?.to_i32()?;
                this.write_scalar(
                    Scalar::from_u64(f.powi(i).to_bits()),
                    dest,
                )?;
            }

            "size_of_val" => {
                let mplace = this.deref_operand(args[0])?;
                let (size, _) = this.size_and_align_of_mplace(mplace)?
                    .expect("size_of_val called on extern type");
                let ptr_size = this.pointer_size();
                this.write_scalar(
                    Scalar::from_uint(size.bytes() as u128, ptr_size),
                    dest,
                )?;
            }

            "min_align_of_val" |
            "align_of_val" => {
                let mplace = this.deref_operand(args[0])?;
                let (_, align) = this.size_and_align_of_mplace(mplace)?
                    .expect("size_of_val called on extern type");
                let ptr_size = this.pointer_size();
                this.write_scalar(
                    Scalar::from_uint(align.bytes(), ptr_size),
                    dest,
                )?;
            }

            "unchecked_div" => {
                let l = this.read_immediate(args[0])?;
                let r = this.read_immediate(args[1])?;
                let rval = r.to_scalar()?.to_bits(args[1].layout.size)?;
                if rval == 0 {
                    throw_ub_format!("Division by 0 in unchecked_div");
                }
                this.binop_ignore_overflow(
                    mir::BinOp::Div,
                    l,
                    r,
                    dest,
                )?;
            }

            "unchecked_rem" => {
                let l = this.read_immediate(args[0])?;
                let r = this.read_immediate(args[1])?;
                let rval = r.to_scalar()?.to_bits(args[1].layout.size)?;
                if rval == 0 {
                    throw_ub_format!("Division by 0 in unchecked_rem");
                }
                this.binop_ignore_overflow(
                    mir::BinOp::Rem,
                    l,
                    r,
                    dest,
                )?;
            }

            "unchecked_add" | "unchecked_sub" | "unchecked_mul" => {
                let l = this.read_immediate(args[0])?;
                let r = this.read_immediate(args[1])?;
                let op = match intrinsic_name {
                    "unchecked_add" => mir::BinOp::Add,
                    "unchecked_sub" => mir::BinOp::Sub,
                    "unchecked_mul" => mir::BinOp::Mul,
                    _ => bug!(),
                };
                let (res, overflowed, _ty) = this.overflowing_binary_op(op, l, r)?;
                if overflowed {
                    throw_ub_format!("Overflowing arithmetic in {}", intrinsic_name);
                }
                this.write_scalar(res, dest)?;
            }

            "uninit" => {
                // Check fast path: we don't want to force an allocation in case the destination is a simple value,
                // but we also do not want to create a new allocation with 0s and then copy that over.
                // FIXME: We do not properly validate in case of ZSTs and when doing it in memory!
                // However, this only affects direct calls of the intrinsic; calls to the stable
                // functions wrapping them do get their validation.
                // FIXME: should we check alignment for ZSTs?
                use crate::ScalarMaybeUndef;
                if !dest.layout.is_zst() {
                    match dest.layout.abi {
                        layout::Abi::Scalar(..) => {
                            let x = ScalarMaybeUndef::Undef;
                            this.write_immediate(Immediate::Scalar(x), dest)?;
                        }
                        layout::Abi::ScalarPair(..) => {
                            let x = ScalarMaybeUndef::Undef;
                            this.write_immediate(Immediate::ScalarPair(x, x), dest)?;
                        }
                        _ => {
                            // Do it in memory
                            let mplace = this.force_allocation(dest)?;
                            mplace.meta.unwrap_none();
                            let ptr = mplace.ptr.to_ptr()?;
                            // We know the return place is in-bounds
                            this.memory
                                .get_mut(ptr.alloc_id)?
                                .mark_definedness(ptr, dest.layout.size, false);
                        }
                    }
                }
            }

            "write_bytes" => {
                let ty = substs.type_at(0);
                let ty_layout = this.layout_of(ty)?;
                let val_byte = this.read_scalar(args[1])?.to_u8()?;
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let count = this.read_scalar(args[2])?.to_usize(this)?;
                let byte_count = ty_layout.size * count;
                this.memory.write_bytes(ptr, iter::repeat(val_byte).take(byte_count.bytes() as usize))?;
            }

            name => throw_unsup_format!("unimplemented intrinsic: {}", name),
        }

        Ok(())
    }
}
