use rustc::mir;
use rustc::traits::Reveal;
use rustc::ty::layout::Layout;
use rustc::ty::{self, Ty};

use rustc_miri::interpret::{EvalResult, Lvalue, LvalueExtra, PrimVal, PrimValKind, Value, Pointer,
                            HasMemory, AccessKind, EvalContext, PtrAndAlign, ValTy};

use helpers::EvalContextExt as HelperEvalContextExt;

pub trait EvalContextExt<'tcx> {
    fn call_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[ValTy<'tcx>],
        dest: Lvalue,
        dest_ty: Ty<'tcx>,
        dest_layout: &'tcx Layout,
        target: mir::BasicBlock,
    ) -> EvalResult<'tcx>;
}

impl<'a, 'tcx> EvalContextExt<'tcx> for EvalContext<'a, 'tcx, super::Evaluator> {
    fn call_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[ValTy<'tcx>],
        dest: Lvalue,
        dest_ty: Ty<'tcx>,
        dest_layout: &'tcx Layout,
        target: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        let substs = instance.substs;

        let intrinsic_name = &self.tcx.item_name(instance.def_id()).as_str()[..];
        match intrinsic_name {
            "align_offset" => {
                // FIXME: return a real value in case the target allocation has an
                // alignment bigger than the one requested
                self.write_primval(dest, PrimVal::Bytes(u128::max_value()), dest_ty)?;
            },

            "add_with_overflow" => {
                self.intrinsic_with_overflow(
                    mir::BinOp::Add,
                    args[0],
                    args[1],
                    dest,
                    dest_ty,
                )?
            }

            "sub_with_overflow" => {
                self.intrinsic_with_overflow(
                    mir::BinOp::Sub,
                    args[0],
                    args[1],
                    dest,
                    dest_ty,
                )?
            }

            "mul_with_overflow" => {
                self.intrinsic_with_overflow(
                    mir::BinOp::Mul,
                    args[0],
                    args[1],
                    dest,
                    dest_ty,
                )?
            }

            "arith_offset" => {
                let offset = self.value_to_primval(args[1])?.to_i128()? as i64;
                let ptr = args[0].into_ptr(&self.memory)?;
                let result_ptr = self.wrapping_pointer_offset(ptr, substs.type_at(0), offset)?;
                self.write_ptr(dest, result_ptr, dest_ty)?;
            }

            "assume" => {
                let cond = self.value_to_primval(args[0])?.to_bool()?;
                if !cond {
                    return err!(AssumptionNotHeld);
                }
            }

            "atomic_load" |
            "atomic_load_relaxed" |
            "atomic_load_acq" |
            "volatile_load" => {
                let ptr = args[0].into_ptr(&self.memory)?;
                let valty = ValTy {
                    value: Value::by_ref(ptr),
                    ty: substs.type_at(0),
                };
                self.write_value(valty, dest)?;
            }

            "atomic_store" |
            "atomic_store_relaxed" |
            "atomic_store_rel" |
            "volatile_store" => {
                let ty = substs.type_at(0);
                let dest = args[0].into_ptr(&self.memory)?;
                self.write_value_to_ptr(args[1].value, dest, ty)?;
            }

            "atomic_fence_acq" => {
                // we are inherently singlethreaded and singlecored, this is a nop
            }

            _ if intrinsic_name.starts_with("atomic_xchg") => {
                let ty = substs.type_at(0);
                let ptr = args[0].into_ptr(&self.memory)?;
                let change = self.value_to_primval(args[1])?;
                let old = self.read_value(ptr, ty)?;
                let old = match old {
                    Value::ByVal(val) => val,
                    Value::ByRef { .. } => bug!("just read the value, can't be byref"),
                    Value::ByValPair(..) => bug!("atomic_xchg doesn't work with nonprimitives"),
                };
                self.write_primval(dest, old, ty)?;
                self.write_primval(
                    Lvalue::from_primval_ptr(ptr),
                    change,
                    ty,
                )?;
            }

            _ if intrinsic_name.starts_with("atomic_cxchg") => {
                let ty = substs.type_at(0);
                let ptr = args[0].into_ptr(&self.memory)?;
                let expect_old = self.value_to_primval(args[1])?;
                let change = self.value_to_primval(args[2])?;
                let old = self.read_value(ptr, ty)?;
                let old = match old {
                    Value::ByVal(val) => val,
                    Value::ByRef { .. } => bug!("just read the value, can't be byref"),
                    Value::ByValPair(..) => bug!("atomic_cxchg doesn't work with nonprimitives"),
                };
                let (val, _) = self.binary_op(mir::BinOp::Eq, old, ty, expect_old, ty)?;
                let dest = self.force_allocation(dest)?.to_ptr()?;
                self.write_pair_to_ptr(old, val, dest, dest_ty)?;
                self.write_primval(
                    Lvalue::from_primval_ptr(ptr),
                    change,
                    ty,
                )?;
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
                let ty = substs.type_at(0);
                let ptr = args[0].into_ptr(&self.memory)?;
                let change = self.value_to_primval(args[1])?;
                let old = self.read_value(ptr, ty)?;
                let old = match old {
                    Value::ByVal(val) => val,
                    Value::ByRef { .. } => bug!("just read the value, can't be byref"),
                    Value::ByValPair(..) => {
                        bug!("atomic_xadd_relaxed doesn't work with nonprimitives")
                    }
                };
                self.write_primval(dest, old, ty)?;
                let op = match intrinsic_name.split('_').nth(1).unwrap() {
                    "or" => mir::BinOp::BitOr,
                    "xor" => mir::BinOp::BitXor,
                    "and" => mir::BinOp::BitAnd,
                    "xadd" => mir::BinOp::Add,
                    "xsub" => mir::BinOp::Sub,
                    _ => bug!(),
                };
                // FIXME: what do atomics do on overflow?
                let (val, _) = self.binary_op(op, old, ty, change, ty)?;
                self.write_primval(Lvalue::from_primval_ptr(ptr), val, ty)?;
            }

            "breakpoint" => unimplemented!(), // halt miri

            "copy" |
            "copy_nonoverlapping" => {
                let elem_ty = substs.type_at(0);
                let elem_size = self.type_size(elem_ty)?.expect("cannot copy unsized value");
                let count = self.value_to_primval(args[2])?.to_u64()?;
                if count * elem_size != 0 {
                    // TODO: We do not even validate alignment for the 0-bytes case.  libstd relies on this in vec::IntoIter::next.
                    // Also see the write_bytes intrinsic.
                    let elem_align = self.type_align(elem_ty)?;
                    let src = args[0].into_ptr(&self.memory)?;
                    let dest = args[1].into_ptr(&self.memory)?;
                    self.memory.copy(
                        src,
                        dest,
                        count * elem_size,
                        elem_align,
                        intrinsic_name.ends_with("_nonoverlapping"),
                    )?;
                }
            }

            "ctpop" | "cttz" | "cttz_nonzero" | "ctlz" | "ctlz_nonzero" | "bswap" => {
                let ty = substs.type_at(0);
                let num = self.value_to_primval(args[0])?.to_bytes()?;
                let kind = self.ty_to_primval_kind(ty)?;
                let num = if intrinsic_name.ends_with("_nonzero") {
                    if num == 0 {
                        return err!(Intrinsic(format!("{} called on 0", intrinsic_name)));
                    }
                    numeric_intrinsic(intrinsic_name.trim_right_matches("_nonzero"), num, kind)?
                } else {
                    numeric_intrinsic(intrinsic_name, num, kind)?
                };
                self.write_primval(dest, num, ty)?;
            }

            "discriminant_value" => {
                let ty = substs.type_at(0);
                let adt_ptr = args[0].into_ptr(&self.memory)?.to_ptr()?;
                let discr_val = self.read_discriminant_value(adt_ptr, ty)?;
                self.write_primval(dest, PrimVal::Bytes(discr_val), dest_ty)?;
            }

            "sinf32" | "fabsf32" | "cosf32" | "sqrtf32" | "expf32" | "exp2f32" | "logf32" |
            "log10f32" | "log2f32" | "floorf32" | "ceilf32" | "truncf32" => {
                let f = self.value_to_primval(args[0])?.to_f32()?;
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
                self.write_primval(dest, PrimVal::from_f32(f), dest_ty)?;
            }

            "sinf64" | "fabsf64" | "cosf64" | "sqrtf64" | "expf64" | "exp2f64" | "logf64" |
            "log10f64" | "log2f64" | "floorf64" | "ceilf64" | "truncf64" => {
                let f = self.value_to_primval(args[0])?.to_f64()?;
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
                self.write_primval(dest, PrimVal::from_f64(f), dest_ty)?;
            }

            "fadd_fast" | "fsub_fast" | "fmul_fast" | "fdiv_fast" | "frem_fast" => {
                let ty = substs.type_at(0);
                let a = self.value_to_primval(args[0])?;
                let b = self.value_to_primval(args[1])?;
                let op = match intrinsic_name {
                    "fadd_fast" => mir::BinOp::Add,
                    "fsub_fast" => mir::BinOp::Sub,
                    "fmul_fast" => mir::BinOp::Mul,
                    "fdiv_fast" => mir::BinOp::Div,
                    "frem_fast" => mir::BinOp::Rem,
                    _ => bug!(),
                };
                let result = self.binary_op(op, a, ty, b, ty)?;
                self.write_primval(dest, result.0, dest_ty)?;
            }

            "likely" | "unlikely" | "forget" => {}

            "init" => {
                let size = self.type_size(dest_ty)?.expect("cannot zero unsized value");
                let init = |this: &mut Self, val: Value| {
                    let zero_val = match val {
                        Value::ByRef(PtrAndAlign { ptr, .. }) => {
                            // These writes have no alignment restriction anyway.
                            this.memory.write_repeat(ptr, 0, size)?;
                            val
                        }
                        // TODO(solson): Revisit this, it's fishy to check for Undef here.
                        Value::ByVal(PrimVal::Undef) => {
                            match this.ty_to_primval_kind(dest_ty) {
                                Ok(_) => Value::ByVal(PrimVal::Bytes(0)),
                                Err(_) => {
                                    let ptr = this.alloc_ptr_with_substs(dest_ty, substs)?;
                                    let ptr = Pointer::from(PrimVal::Ptr(ptr));
                                    this.memory.write_repeat(ptr, 0, size)?;
                                    Value::by_ref(ptr)
                                }
                            }
                        }
                        Value::ByVal(_) => Value::ByVal(PrimVal::Bytes(0)),
                        Value::ByValPair(..) => {
                            Value::ByValPair(PrimVal::Bytes(0), PrimVal::Bytes(0))
                        }
                    };
                    Ok(zero_val)
                };
                match dest {
                    Lvalue::Local { frame, local } => self.modify_local(frame, local, init)?,
                    Lvalue::Ptr {
                        ptr: PtrAndAlign { ptr, aligned: true },
                        extra: LvalueExtra::None,
                    } => self.memory.write_repeat(ptr, 0, size)?,
                    Lvalue::Ptr { .. } => {
                        bug!("init intrinsic tried to write to fat or unaligned ptr target")
                    }
                }
            }

            "min_align_of" => {
                let elem_ty = substs.type_at(0);
                let elem_align = self.type_align(elem_ty)?;
                let align_val = PrimVal::from_u128(elem_align as u128);
                self.write_primval(dest, align_val, dest_ty)?;
            }

            "pref_align_of" => {
                let ty = substs.type_at(0);
                let layout = self.type_layout(ty)?;
                let align = layout.align(&self.tcx.data_layout).pref();
                let align_val = PrimVal::from_u128(align as u128);
                self.write_primval(dest, align_val, dest_ty)?;
            }

            "move_val_init" => {
                let ty = substs.type_at(0);
                let ptr = args[0].into_ptr(&self.memory)?;
                self.write_value_to_ptr(args[1].value, ptr, ty)?;
            }

            "needs_drop" => {
                let ty = substs.type_at(0);
                let env = ty::ParamEnv::empty(Reveal::All);
                let needs_drop = ty.needs_drop(self.tcx, env);
                self.write_primval(
                    dest,
                    PrimVal::from_bool(needs_drop),
                    dest_ty,
                )?;
            }

            "offset" => {
                let offset = self.value_to_primval(args[1])?.to_i128()? as i64;
                let ptr = args[0].into_ptr(&self.memory)?;
                let result_ptr = self.pointer_offset(ptr, substs.type_at(0), offset)?;
                self.write_ptr(dest, result_ptr, dest_ty)?;
            }

            "overflowing_sub" => {
                self.intrinsic_overflowing(
                    mir::BinOp::Sub,
                    args[0],
                    args[1],
                    dest,
                    dest_ty,
                )?;
            }

            "overflowing_mul" => {
                self.intrinsic_overflowing(
                    mir::BinOp::Mul,
                    args[0],
                    args[1],
                    dest,
                    dest_ty,
                )?;
            }

            "overflowing_add" => {
                self.intrinsic_overflowing(
                    mir::BinOp::Add,
                    args[0],
                    args[1],
                    dest,
                    dest_ty,
                )?;
            }

            "powf32" => {
                let f = self.value_to_primval(args[0])?.to_f32()?;
                let f2 = self.value_to_primval(args[1])?.to_f32()?;
                self.write_primval(
                    dest,
                    PrimVal::from_f32(f.powf(f2)),
                    dest_ty,
                )?;
            }

            "powf64" => {
                let f = self.value_to_primval(args[0])?.to_f64()?;
                let f2 = self.value_to_primval(args[1])?.to_f64()?;
                self.write_primval(
                    dest,
                    PrimVal::from_f64(f.powf(f2)),
                    dest_ty,
                )?;
            }

            "fmaf32" => {
                let a = self.value_to_primval(args[0])?.to_f32()?;
                let b = self.value_to_primval(args[1])?.to_f32()?;
                let c = self.value_to_primval(args[2])?.to_f32()?;
                self.write_primval(
                    dest,
                    PrimVal::from_f32(a * b + c),
                    dest_ty,
                )?;
            }

            "fmaf64" => {
                let a = self.value_to_primval(args[0])?.to_f64()?;
                let b = self.value_to_primval(args[1])?.to_f64()?;
                let c = self.value_to_primval(args[2])?.to_f64()?;
                self.write_primval(
                    dest,
                    PrimVal::from_f64(a * b + c),
                    dest_ty,
                )?;
            }

            "powif32" => {
                let f = self.value_to_primval(args[0])?.to_f32()?;
                let i = self.value_to_primval(args[1])?.to_i128()?;
                self.write_primval(
                    dest,
                    PrimVal::from_f32(f.powi(i as i32)),
                    dest_ty,
                )?;
            }

            "powif64" => {
                let f = self.value_to_primval(args[0])?.to_f64()?;
                let i = self.value_to_primval(args[1])?.to_i128()?;
                self.write_primval(
                    dest,
                    PrimVal::from_f64(f.powi(i as i32)),
                    dest_ty,
                )?;
            }

            "size_of" => {
                let ty = substs.type_at(0);
                let size = self.type_size(ty)?.expect(
                    "size_of intrinsic called on unsized value",
                ) as u128;
                self.write_primval(dest, PrimVal::from_u128(size), dest_ty)?;
            }

            "size_of_val" => {
                let ty = substs.type_at(0);
                let (size, _) = self.size_and_align_of_dst(ty, args[0].value)?;
                self.write_primval(
                    dest,
                    PrimVal::from_u128(size as u128),
                    dest_ty,
                )?;
            }

            "min_align_of_val" |
            "align_of_val" => {
                let ty = substs.type_at(0);
                let (_, align) = self.size_and_align_of_dst(ty, args[0].value)?;
                self.write_primval(
                    dest,
                    PrimVal::from_u128(align as u128),
                    dest_ty,
                )?;
            }

            "type_name" => {
                let ty = substs.type_at(0);
                let ty_name = ty.to_string();
                let value = self.str_to_value(&ty_name)?;
                self.write_value(ValTy { value, ty: dest_ty }, dest)?;
            }
            "type_id" => {
                let ty = substs.type_at(0);
                let n = self.tcx.type_id_hash(ty);
                self.write_primval(dest, PrimVal::Bytes(n as u128), dest_ty)?;
            }

            "transmute" => {
                let src_ty = substs.type_at(0);
                let ptr = self.force_allocation(dest)?.to_ptr()?;
                self.write_maybe_aligned_mut(
                    /*aligned*/
                    false,
                    |ectx| {
                        ectx.write_value_to_ptr(args[0].value, ptr.into(), src_ty)
                    },
                )?;
            }

            "unchecked_shl" => {
                let bits = self.type_size(dest_ty)?.expect(
                    "intrinsic can't be called on unsized type",
                ) as u128 * 8;
                let rhs = self.value_to_primval(args[1])?
                    .to_bytes()?;
                if rhs >= bits {
                    return err!(Intrinsic(
                        format!("Overflowing shift by {} in unchecked_shl", rhs),
                    ));
                }
                self.intrinsic_overflowing(
                    mir::BinOp::Shl,
                    args[0],
                    args[1],
                    dest,
                    dest_ty,
                )?;
            }

            "unchecked_shr" => {
                let bits = self.type_size(dest_ty)?.expect(
                    "intrinsic can't be called on unsized type",
                ) as u128 * 8;
                let rhs = self.value_to_primval(args[1])?
                    .to_bytes()?;
                if rhs >= bits {
                    return err!(Intrinsic(
                        format!("Overflowing shift by {} in unchecked_shr", rhs),
                    ));
                }
                self.intrinsic_overflowing(
                    mir::BinOp::Shr,
                    args[0],
                    args[1],
                    dest,
                    dest_ty,
                )?;
            }

            "unchecked_div" => {
                let rhs = self.value_to_primval(args[1])?
                    .to_bytes()?;
                if rhs == 0 {
                    return err!(Intrinsic(format!("Division by 0 in unchecked_div")));
                }
                self.intrinsic_overflowing(
                    mir::BinOp::Div,
                    args[0],
                    args[1],
                    dest,
                    dest_ty,
                )?;
            }

            "unchecked_rem" => {
                let rhs = self.value_to_primval(args[1])?
                    .to_bytes()?;
                if rhs == 0 {
                    return err!(Intrinsic(format!("Division by 0 in unchecked_rem")));
                }
                self.intrinsic_overflowing(
                    mir::BinOp::Rem,
                    args[0],
                    args[1],
                    dest,
                    dest_ty,
                )?;
            }

            "uninit" => {
                let size = dest_layout.size(&self.tcx.data_layout).bytes();
                let uninit = |this: &mut Self, val: Value| match val {
                    Value::ByRef(PtrAndAlign { ptr, .. }) => {
                        this.memory.mark_definedness(ptr, size, false)?;
                        Ok(val)
                    }
                    _ => Ok(Value::ByVal(PrimVal::Undef)),
                };
                match dest {
                    Lvalue::Local { frame, local } => self.modify_local(frame, local, uninit)?,
                    Lvalue::Ptr {
                        ptr: PtrAndAlign { ptr, aligned: true },
                        extra: LvalueExtra::None,
                    } => self.memory.mark_definedness(ptr, size, false)?,
                    Lvalue::Ptr { .. } => {
                        bug!("uninit intrinsic tried to write to fat or unaligned ptr target")
                    }
                }
            }

            "write_bytes" => {
                let ty = substs.type_at(0);
                let ty_align = self.type_align(ty)?;
                let val_byte = self.value_to_primval(args[1])?.to_u128()? as u8;
                let size = self.type_size(ty)?.expect(
                    "write_bytes() type must be sized",
                );
                let ptr = args[0].into_ptr(&self.memory)?;
                let count = self.value_to_primval(args[2])?.to_u64()?;
                if count > 0 {
                    // HashMap relies on write_bytes on a NULL ptr with count == 0 to work
                    // TODO: Should we, at least, validate the alignment? (Also see the copy intrinsic)
                    self.memory.check_align(ptr, ty_align, Some(AccessKind::Write))?;
                    self.memory.write_repeat(ptr, val_byte, size * count)?;
                }
            }

            name => return err!(Unimplemented(format!("unimplemented intrinsic: {}", name))),
        }

        self.goto_block(target);

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        Ok(())
    }
}

fn numeric_intrinsic<'tcx>(
    name: &str,
    bytes: u128,
    kind: PrimValKind,
) -> EvalResult<'tcx, PrimVal> {
    macro_rules! integer_intrinsic {
        ($method:ident) => ({
            use rustc_miri::interpret::PrimValKind::*;
            let result_bytes = match kind {
                I8 => (bytes as i8).$method() as u128,
                U8 => (bytes as u8).$method() as u128,
                I16 => (bytes as i16).$method() as u128,
                U16 => (bytes as u16).$method() as u128,
                I32 => (bytes as i32).$method() as u128,
                U32 => (bytes as u32).$method() as u128,
                I64 => (bytes as i64).$method() as u128,
                U64 => (bytes as u64).$method() as u128,
                I128 => (bytes as i128).$method() as u128,
                U128 => bytes.$method() as u128,
                _ => bug!("invalid `{}` argument: {:?}", name, bytes),
            };

            PrimVal::Bytes(result_bytes)
        });
    }

    let result_val = match name {
        "bswap" => integer_intrinsic!(swap_bytes),
        "ctlz" => integer_intrinsic!(leading_zeros),
        "ctpop" => integer_intrinsic!(count_ones),
        "cttz" => integer_intrinsic!(trailing_zeros),
        _ => bug!("not a numeric intrinsic: {}", name),
    };

    Ok(result_val)
}
