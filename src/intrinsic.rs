use rustc::mir;
use rustc::ty::layout::{self, LayoutOf, Size};
use rustc::ty;

use rustc::mir::interpret::{EvalResult, Scalar, ScalarMaybeUndef, PointerArithmetic};
use rustc_mir::interpret::{
    PlaceTy, EvalContext, OpTy, Value
};

use super::{FalibleScalarExt, OperatorEvalContextExt};

pub trait EvalContextExt<'tcx> {
    fn call_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: PlaceTy<'tcx>,
    ) -> EvalResult<'tcx>;
}

impl<'a, 'mir, 'tcx> EvalContextExt<'tcx> for EvalContext<'a, 'mir, 'tcx, super::Evaluator<'tcx>> {
    fn call_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: PlaceTy<'tcx>,
    ) -> EvalResult<'tcx> {
        if self.emulate_intrinsic(instance, args, dest)? {
            return Ok(());
        }

        let substs = instance.substs;

        let intrinsic_name = &self.tcx.item_name(instance.def_id()).as_str()[..];
        match intrinsic_name {
            "arith_offset" => {
                let offset = self.read_scalar(args[1])?.to_isize(&self)?;
                let ptr = self.read_scalar(args[0])?.not_undef()?;

                let pointee_ty = substs.type_at(0);
                let pointee_size = self.layout_of(pointee_ty)?.size.bytes() as i64;
                let offset = offset.overflowing_mul(pointee_size).0;
                let result_ptr = ptr.ptr_wrapping_signed_offset(offset, &self);
                self.write_scalar(result_ptr, dest)?;
            }

            "assume" => {
                let cond = self.read_scalar(args[0])?.to_bool()?;
                if !cond {
                    return err!(AssumptionNotHeld);
                }
            }

            "atomic_load" |
            "atomic_load_relaxed" |
            "atomic_load_acq" |
            "volatile_load" => {
                let ptr = self.ref_to_mplace(self.read_value(args[0])?)?;
                let val = self.read_scalar(ptr.into())?; // make sure it fits into a scalar; otherwise it cannot be atomic
                self.write_scalar(val, dest)?;
            }

            "atomic_store" |
            "atomic_store_relaxed" |
            "atomic_store_rel" |
            "volatile_store" => {
                let ptr = self.ref_to_mplace(self.read_value(args[0])?)?;
                let val = self.read_scalar(args[1])?; // make sure it fits into a scalar; otherwise it cannot be atomic
                self.write_scalar(val, ptr.into())?;
            }

            "atomic_fence_acq" => {
                // we are inherently singlethreaded and singlecored, this is a nop
            }

            _ if intrinsic_name.starts_with("atomic_xchg") => {
                let ptr = self.ref_to_mplace(self.read_value(args[0])?)?;
                let new = self.read_scalar(args[1])?;
                let old = self.read_scalar(ptr.into())?;
                self.write_scalar(old, dest)?; // old value is returned
                self.write_scalar(new, ptr.into())?;
            }

            _ if intrinsic_name.starts_with("atomic_cxchg") => {
                let ptr = self.ref_to_mplace(self.read_value(args[0])?)?;
                let expect_old = self.read_value(args[1])?; // read as value for the sake of `binary_op_val()`
                let new = self.read_scalar(args[2])?;
                let old = self.read_value(ptr.into())?; // read as value for the sake of `binary_op_val()`
                // binary_op_val will bail if either of them is not a scalar
                let (eq, _) = self.binary_op_val(mir::BinOp::Eq, old, expect_old)?;
                let res = Value::ScalarPair(old.to_scalar_or_undef(), eq.into());
                self.write_value(res, dest)?; // old value is returned
                // update ptr depending on comparison
                if eq.to_bool()? {
                    self.write_scalar(new, ptr.into())?;
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
                let ptr = self.ref_to_mplace(self.read_value(args[0])?)?;
                let rhs = self.read_value(args[1])?;
                let old = self.read_value(ptr.into())?;
                self.write_value(*old, dest)?; // old value is returned
                let op = match intrinsic_name.split('_').nth(1).unwrap() {
                    "or" => mir::BinOp::BitOr,
                    "xor" => mir::BinOp::BitXor,
                    "and" => mir::BinOp::BitAnd,
                    "xadd" => mir::BinOp::Add,
                    "xsub" => mir::BinOp::Sub,
                    _ => bug!(),
                };
                // FIXME: what do atomics do on overflow?
                self.binop_ignore_overflow(op, old, rhs, ptr.into())?;
            }

            "breakpoint" => unimplemented!(), // halt miri

            "copy" |
            "copy_nonoverlapping" => {
                let elem_ty = substs.type_at(0);
                let elem_layout = self.layout_of(elem_ty)?;
                let elem_size = elem_layout.size.bytes();
                let count = self.read_scalar(args[2])?.to_usize(&self)?;
                let elem_align = elem_layout.align;
                let src = self.read_scalar(args[0])?.not_undef()?;
                let dest = self.read_scalar(args[1])?.not_undef()?;
                self.memory.copy(
                    src,
                    elem_align,
                    dest,
                    elem_align,
                    Size::from_bytes(count * elem_size),
                    intrinsic_name.ends_with("_nonoverlapping"),
                )?;
            }

            "discriminant_value" => {
                let place = self.ref_to_mplace(self.read_value(args[0])?)?;
                let discr_val = self.read_discriminant(place.into())?.0;
                self.write_scalar(Scalar::from_uint(discr_val, dest.layout.size), dest)?;
            }

            "sinf32" | "fabsf32" | "cosf32" | "sqrtf32" | "expf32" | "exp2f32" | "logf32" |
            "log10f32" | "log2f32" | "floorf32" | "ceilf32" | "truncf32" => {
                let f = self.read_scalar(args[0])?.to_f32()?;
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
                    _ => bug!(),
                };
                self.write_scalar(Scalar::from_f32(f), dest)?;
            }

            "sinf64" | "fabsf64" | "cosf64" | "sqrtf64" | "expf64" | "exp2f64" | "logf64" |
            "log10f64" | "log2f64" | "floorf64" | "ceilf64" | "truncf64" => {
                let f = self.read_scalar(args[0])?.to_f64()?;
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
                    _ => bug!(),
                };
                self.write_scalar(Scalar::from_f64(f), dest)?;
            }

            "fadd_fast" | "fsub_fast" | "fmul_fast" | "fdiv_fast" | "frem_fast" => {
                let a = self.read_value(args[0])?;
                let b = self.read_value(args[1])?;
                let op = match intrinsic_name {
                    "fadd_fast" => mir::BinOp::Add,
                    "fsub_fast" => mir::BinOp::Sub,
                    "fmul_fast" => mir::BinOp::Mul,
                    "fdiv_fast" => mir::BinOp::Div,
                    "frem_fast" => mir::BinOp::Rem,
                    _ => bug!(),
                };
                self.binop_ignore_overflow(op, a, b, dest)?;
            }

            "exact_div" => {
                // Performs an exact division, resulting in undefined behavior where
                // `x % y != 0` or `y == 0` or `x == T::min_value() && y == -1`
                let a = self.read_value(args[0])?;
                let b = self.read_value(args[1])?;
                // check x % y != 0
                if !self.binary_op_val(mir::BinOp::Rem, a, b)?.0.is_null() {
                    return err!(ValidationFailure(format!("exact_div: {:?} cannot be divided by {:?}", a, b)));
                }
                self.binop_ignore_overflow(mir::BinOp::Div, a, b, dest)?;
            },

            "likely" | "unlikely" | "forget" => {}

            "init" => {
                // Check fast path: we don't want to force an allocation in case the destination is a simple value,
                // but we also do not want to create a new allocation with 0s and then copy that over.
                if !dest.layout.is_zst() { // notzhing to do for ZST
                    match dest.layout.abi {
                        layout::Abi::Scalar(ref s) => {
                            let x = Scalar::from_int(0, s.value.size(&self));
                            self.write_value(Value::Scalar(x.into()), dest)?;
                        }
                        layout::Abi::ScalarPair(ref s1, ref s2) => {
                            let x = Scalar::from_int(0, s1.value.size(&self));
                            let y = Scalar::from_int(0, s2.value.size(&self));
                            self.write_value(Value::ScalarPair(x.into(), y.into()), dest)?;
                        }
                        _ => {
                            // Do it in memory
                            let mplace = self.force_allocation(dest)?;
                            assert!(mplace.extra.is_none());
                            self.memory.write_repeat(mplace.ptr, 0, dest.layout.size)?;
                        }
                    }
                }
            }

            "pref_align_of" => {
                let ty = substs.type_at(0);
                let layout = self.layout_of(ty)?;
                let align = layout.align.pref();
                let ptr_size = self.pointer_size();
                let align_val = Scalar::from_uint(align as u128, ptr_size);
                self.write_scalar(align_val, dest)?;
            }

            "move_val_init" => {
                let ptr = self.ref_to_mplace(self.read_value(args[0])?)?;
                self.copy_op(args[1], ptr.into())?;
            }

            "offset" => {
                let offset = self.read_scalar(args[1])?.to_isize(&self)?;
                let ptr = self.read_scalar(args[0])?.not_undef()?;
                let result_ptr = self.pointer_offset_inbounds(ptr, substs.type_at(0), offset)?;
                self.write_scalar(result_ptr, dest)?;
            }

            "powf32" => {
                let f = self.read_scalar(args[0])?.to_f32()?;
                let f2 = self.read_scalar(args[1])?.to_f32()?;
                self.write_scalar(
                    Scalar::from_f32(f.powf(f2)),
                    dest,
                )?;
            }

            "powf64" => {
                let f = self.read_scalar(args[0])?.to_f64()?;
                let f2 = self.read_scalar(args[1])?.to_f64()?;
                self.write_scalar(
                    Scalar::from_f64(f.powf(f2)),
                    dest,
                )?;
            }

            "fmaf32" => {
                let a = self.read_scalar(args[0])?.to_f32()?;
                let b = self.read_scalar(args[1])?.to_f32()?;
                let c = self.read_scalar(args[2])?.to_f32()?;
                self.write_scalar(
                    Scalar::from_f32(a * b + c),
                    dest,
                )?;
            }

            "fmaf64" => {
                let a = self.read_scalar(args[0])?.to_f64()?;
                let b = self.read_scalar(args[1])?.to_f64()?;
                let c = self.read_scalar(args[2])?.to_f64()?;
                self.write_scalar(
                    Scalar::from_f64(a * b + c),
                    dest,
                )?;
            }

            "powif32" => {
                let f = self.read_scalar(args[0])?.to_f32()?;
                let i = self.read_scalar(args[1])?.to_i32()?;
                self.write_scalar(
                    Scalar::from_f32(f.powi(i)),
                    dest,
                )?;
            }

            "powif64" => {
                let f = self.read_scalar(args[0])?.to_f64()?;
                let i = self.read_scalar(args[1])?.to_i32()?;
                self.write_scalar(
                    Scalar::from_f64(f.powi(i)),
                    dest,
                )?;
            }

            "size_of_val" => {
                let mplace = self.ref_to_mplace(self.read_value(args[0])?)?;
                let (size, _) = self.size_and_align_of_mplace(mplace)?;
                let ptr_size = self.pointer_size();
                self.write_scalar(
                    Scalar::from_uint(size.bytes() as u128, ptr_size),
                    dest,
                )?;
            }

            "min_align_of_val" |
            "align_of_val" => {
                let mplace = self.ref_to_mplace(self.read_value(args[0])?)?;
                let (_, align) = self.size_and_align_of_mplace(mplace)?;
                let ptr_size = self.pointer_size();
                self.write_scalar(
                    Scalar::from_uint(align.abi(), ptr_size),
                    dest,
                )?;
            }

            "type_name" => {
                let ty = substs.type_at(0);
                let ty_name = ty.to_string();
                let value = self.str_to_value(&ty_name)?;
                self.write_value(value, dest)?;
            }

            "unchecked_div" => {
                let l = self.read_value(args[0])?;
                let r = self.read_value(args[1])?;
                let rval = r.to_scalar()?.to_bytes()?;
                if rval == 0 {
                    return err!(Intrinsic(format!("Division by 0 in unchecked_div")));
                }
                self.binop_ignore_overflow(
                    mir::BinOp::Div,
                    l,
                    r,
                    dest,
                )?;
            }

            "unchecked_rem" => {
                let l = self.read_value(args[0])?;
                let r = self.read_value(args[1])?;
                let rval = r.to_scalar()?.to_bytes()?;
                if rval == 0 {
                    return err!(Intrinsic(format!("Division by 0 in unchecked_rem")));
                }
                self.binop_ignore_overflow(
                    mir::BinOp::Rem,
                    l,
                    r,
                    dest,
                )?;
            }

            "uninit" => {
                // Check fast path: we don't want to force an allocation in case the destination is a simple value,
                // but we also do not want to create a new allocation with 0s and then copy that over.
                if !dest.layout.is_zst() { // nothing to do for ZST
                    match dest.layout.abi {
                        layout::Abi::Scalar(..) => {
                            let x = ScalarMaybeUndef::Undef;
                            self.write_value(Value::Scalar(x), dest)?;
                        }
                        layout::Abi::ScalarPair(..) => {
                            let x = ScalarMaybeUndef::Undef;
                            self.write_value(Value::ScalarPair(x, x), dest)?;
                        }
                        _ => {
                            // Do it in memory
                            let mplace = self.force_allocation(dest)?;
                            assert!(mplace.extra.is_none());
                            self.memory.mark_definedness(mplace.ptr.to_ptr()?, dest.layout.size, false)?;
                        }
                    }
                }
            }

            "write_bytes" => {
                let ty = substs.type_at(0);
                let ty_layout = self.layout_of(ty)?;
                let val_byte = self.read_scalar(args[1])?.to_u8()?;
                let ptr = self.read_scalar(args[0])?.not_undef()?;
                let count = self.read_scalar(args[2])?.to_usize(&self)?;
                self.memory.check_align(ptr, ty_layout.align)?;
                self.memory.write_repeat(ptr, val_byte, ty_layout.size * count)?;
            }

            name => return err!(Unimplemented(format!("unimplemented intrinsic: {}", name))),
        }

        Ok(())
    }
}
