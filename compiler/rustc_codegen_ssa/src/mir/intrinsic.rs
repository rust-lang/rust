use super::operand::{OperandRef, OperandValue};
use super::place::PlaceRef;
use super::FunctionCx;
use crate::common::{span_invalid_monomorphization_error, IntPredicate};
use crate::glue;
use crate::traits::*;
use crate::MemFlags;

use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::{sym, Span};
use rustc_target::abi::call::{FnAbi, PassMode};

fn copy_intrinsic<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    allow_overlap: bool,
    volatile: bool,
    ty: Ty<'tcx>,
    dst: Bx::Value,
    src: Bx::Value,
    count: Bx::Value,
) {
    let layout = bx.layout_of(ty);
    let size = layout.size;
    let align = layout.align.abi;
    let size = bx.mul(bx.const_usize(size.bytes()), count);
    let flags = if volatile { MemFlags::VOLATILE } else { MemFlags::empty() };
    if allow_overlap {
        bx.memmove(dst, align, src, align, size, flags);
    } else {
        bx.memcpy(dst, align, src, align, size, flags);
    }
}

fn memset_intrinsic<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    volatile: bool,
    ty: Ty<'tcx>,
    dst: Bx::Value,
    val: Bx::Value,
    count: Bx::Value,
) {
    let layout = bx.layout_of(ty);
    let size = layout.size;
    let align = layout.align.abi;
    let size = bx.mul(bx.const_usize(size.bytes()), count);
    let flags = if volatile { MemFlags::VOLATILE } else { MemFlags::empty() };
    bx.memset(dst, val, size, align, flags);
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn codegen_intrinsic_call(
        bx: &mut Bx,
        instance: ty::Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OperandRef<'tcx, Bx::Value>],
        llresult: Bx::Value,
        span: Span,
    ) {
        let callee_ty = instance.ty(bx.tcx(), ty::ParamEnv::reveal_all());

        let (def_id, substs) = match *callee_ty.kind() {
            ty::FnDef(def_id, substs) => (def_id, substs),
            _ => bug!("expected fn item type, found {}", callee_ty),
        };

        let sig = callee_ty.fn_sig(bx.tcx());
        let sig = bx.tcx().normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), sig);
        let arg_tys = sig.inputs();
        let ret_ty = sig.output();
        let name = bx.tcx().item_name(def_id);
        let name_str = &*name.as_str();

        let llret_ty = bx.backend_type(bx.layout_of(ret_ty));
        let result = PlaceRef::new_sized(llresult, fn_abi.ret.layout);

        let llval = match name {
            sym::assume => {
                bx.assume(args[0].immediate());
                return;
            }
            sym::abort => {
                bx.abort();
                return;
            }

            sym::va_start => bx.va_start(args[0].immediate()),
            sym::va_end => bx.va_end(args[0].immediate()),
            sym::size_of_val => {
                let tp_ty = substs.type_at(0);
                if let OperandValue::Pair(_, meta) = args[0].val {
                    let (llsize, _) = glue::size_and_align_of_dst(bx, tp_ty, Some(meta));
                    llsize
                } else {
                    bx.const_usize(bx.layout_of(tp_ty).size.bytes())
                }
            }
            sym::min_align_of_val => {
                let tp_ty = substs.type_at(0);
                if let OperandValue::Pair(_, meta) = args[0].val {
                    let (_, llalign) = glue::size_and_align_of_dst(bx, tp_ty, Some(meta));
                    llalign
                } else {
                    bx.const_usize(bx.layout_of(tp_ty).align.abi.bytes())
                }
            }
            sym::pref_align_of
            | sym::min_align_of
            | sym::needs_drop
            | sym::type_id
            | sym::type_name
            | sym::variant_count => {
                let value = bx
                    .tcx()
                    .const_eval_instance(ty::ParamEnv::reveal_all(), instance, None)
                    .unwrap();
                OperandRef::from_const(bx, value, ret_ty).immediate_or_packed_pair(bx)
            }
            sym::offset => {
                let ptr = args[0].immediate();
                let offset = args[1].immediate();
                bx.inbounds_gep(ptr, &[offset])
            }
            sym::arith_offset => {
                let ptr = args[0].immediate();
                let offset = args[1].immediate();
                bx.gep(ptr, &[offset])
            }

            sym::copy_nonoverlapping => {
                copy_intrinsic(
                    bx,
                    false,
                    false,
                    substs.type_at(0),
                    args[1].immediate(),
                    args[0].immediate(),
                    args[2].immediate(),
                );
                return;
            }
            sym::copy => {
                copy_intrinsic(
                    bx,
                    true,
                    false,
                    substs.type_at(0),
                    args[1].immediate(),
                    args[0].immediate(),
                    args[2].immediate(),
                );
                return;
            }
            sym::write_bytes => {
                memset_intrinsic(
                    bx,
                    false,
                    substs.type_at(0),
                    args[0].immediate(),
                    args[1].immediate(),
                    args[2].immediate(),
                );
                return;
            }

            sym::volatile_copy_nonoverlapping_memory => {
                copy_intrinsic(
                    bx,
                    false,
                    true,
                    substs.type_at(0),
                    args[0].immediate(),
                    args[1].immediate(),
                    args[2].immediate(),
                );
                return;
            }
            sym::volatile_copy_memory => {
                copy_intrinsic(
                    bx,
                    true,
                    true,
                    substs.type_at(0),
                    args[0].immediate(),
                    args[1].immediate(),
                    args[2].immediate(),
                );
                return;
            }
            sym::volatile_set_memory => {
                memset_intrinsic(
                    bx,
                    true,
                    substs.type_at(0),
                    args[0].immediate(),
                    args[1].immediate(),
                    args[2].immediate(),
                );
                return;
            }
            sym::volatile_store => {
                let dst = args[0].deref(bx.cx());
                args[1].val.volatile_store(bx, dst);
                return;
            }
            sym::unaligned_volatile_store => {
                let dst = args[0].deref(bx.cx());
                args[1].val.unaligned_volatile_store(bx, dst);
                return;
            }
            sym::add_with_overflow
            | sym::sub_with_overflow
            | sym::mul_with_overflow
            | sym::unchecked_div
            | sym::unchecked_rem
            | sym::unchecked_shl
            | sym::unchecked_shr
            | sym::unchecked_add
            | sym::unchecked_sub
            | sym::unchecked_mul
            | sym::exact_div => {
                let ty = arg_tys[0];
                match int_type_width_signed(ty, bx.tcx()) {
                    Some((_width, signed)) => match name {
                        sym::add_with_overflow
                        | sym::sub_with_overflow
                        | sym::mul_with_overflow => {
                            let op = match name {
                                sym::add_with_overflow => OverflowOp::Add,
                                sym::sub_with_overflow => OverflowOp::Sub,
                                sym::mul_with_overflow => OverflowOp::Mul,
                                _ => bug!(),
                            };
                            let (val, overflow) =
                                bx.checked_binop(op, ty, args[0].immediate(), args[1].immediate());
                            // Convert `i1` to a `bool`, and write it to the out parameter
                            let val = bx.from_immediate(val);
                            let overflow = bx.from_immediate(overflow);

                            let dest = result.project_field(bx, 0);
                            bx.store(val, dest.llval, dest.align);
                            let dest = result.project_field(bx, 1);
                            bx.store(overflow, dest.llval, dest.align);

                            return;
                        }
                        sym::exact_div => {
                            if signed {
                                bx.exactsdiv(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.exactudiv(args[0].immediate(), args[1].immediate())
                            }
                        }
                        sym::unchecked_div => {
                            if signed {
                                bx.sdiv(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.udiv(args[0].immediate(), args[1].immediate())
                            }
                        }
                        sym::unchecked_rem => {
                            if signed {
                                bx.srem(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.urem(args[0].immediate(), args[1].immediate())
                            }
                        }
                        sym::unchecked_shl => bx.shl(args[0].immediate(), args[1].immediate()),
                        sym::unchecked_shr => {
                            if signed {
                                bx.ashr(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.lshr(args[0].immediate(), args[1].immediate())
                            }
                        }
                        sym::unchecked_add => {
                            if signed {
                                bx.unchecked_sadd(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.unchecked_uadd(args[0].immediate(), args[1].immediate())
                            }
                        }
                        sym::unchecked_sub => {
                            if signed {
                                bx.unchecked_ssub(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.unchecked_usub(args[0].immediate(), args[1].immediate())
                            }
                        }
                        sym::unchecked_mul => {
                            if signed {
                                bx.unchecked_smul(args[0].immediate(), args[1].immediate())
                            } else {
                                bx.unchecked_umul(args[0].immediate(), args[1].immediate())
                            }
                        }
                        _ => bug!(),
                    },
                    None => {
                        span_invalid_monomorphization_error(
                            bx.tcx().sess,
                            span,
                            &format!(
                                "invalid monomorphization of `{}` intrinsic: \
                                      expected basic integer type, found `{}`",
                                name, ty
                            ),
                        );
                        return;
                    }
                }
            }
            sym::fadd_fast | sym::fsub_fast | sym::fmul_fast | sym::fdiv_fast | sym::frem_fast => {
                match float_type_width(arg_tys[0]) {
                    Some(_width) => match name {
                        sym::fadd_fast => bx.fadd_fast(args[0].immediate(), args[1].immediate()),
                        sym::fsub_fast => bx.fsub_fast(args[0].immediate(), args[1].immediate()),
                        sym::fmul_fast => bx.fmul_fast(args[0].immediate(), args[1].immediate()),
                        sym::fdiv_fast => bx.fdiv_fast(args[0].immediate(), args[1].immediate()),
                        sym::frem_fast => bx.frem_fast(args[0].immediate(), args[1].immediate()),
                        _ => bug!(),
                    },
                    None => {
                        span_invalid_monomorphization_error(
                            bx.tcx().sess,
                            span,
                            &format!(
                                "invalid monomorphization of `{}` intrinsic: \
                                      expected basic float type, found `{}`",
                                name, arg_tys[0]
                            ),
                        );
                        return;
                    }
                }
            }

            sym::float_to_int_unchecked => {
                if float_type_width(arg_tys[0]).is_none() {
                    span_invalid_monomorphization_error(
                        bx.tcx().sess,
                        span,
                        &format!(
                            "invalid monomorphization of `float_to_int_unchecked` \
                                  intrinsic: expected basic float type, \
                                  found `{}`",
                            arg_tys[0]
                        ),
                    );
                    return;
                }
                let (_width, signed) = match int_type_width_signed(ret_ty, bx.tcx()) {
                    Some(pair) => pair,
                    None => {
                        span_invalid_monomorphization_error(
                            bx.tcx().sess,
                            span,
                            &format!(
                                "invalid monomorphization of `float_to_int_unchecked` \
                                      intrinsic:  expected basic integer type, \
                                      found `{}`",
                                ret_ty
                            ),
                        );
                        return;
                    }
                };
                if signed {
                    bx.fptosi(args[0].immediate(), llret_ty)
                } else {
                    bx.fptoui(args[0].immediate(), llret_ty)
                }
            }

            sym::discriminant_value => {
                if ret_ty.is_integral() {
                    args[0].deref(bx.cx()).codegen_get_discr(bx, ret_ty)
                } else {
                    span_bug!(span, "Invalid discriminant type for `{:?}`", arg_tys[0])
                }
            }

            // This requires that atomic intrinsics follow a specific naming pattern:
            // "atomic_<operation>[_<ordering>]", and no ordering means SeqCst
            name if name_str.starts_with("atomic_") => {
                use crate::common::AtomicOrdering::*;
                use crate::common::{AtomicRmwBinOp, SynchronizationScope};

                let split: Vec<&str> = name_str.split('_').collect();

                let is_cxchg = split[1] == "cxchg" || split[1] == "cxchgweak";
                let (order, failorder) = match split.len() {
                    2 => (SequentiallyConsistent, SequentiallyConsistent),
                    3 => match split[2] {
                        "unordered" => (Unordered, Unordered),
                        "relaxed" => (Monotonic, Monotonic),
                        "acq" => (Acquire, Acquire),
                        "rel" => (Release, Monotonic),
                        "acqrel" => (AcquireRelease, Acquire),
                        "failrelaxed" if is_cxchg => (SequentiallyConsistent, Monotonic),
                        "failacq" if is_cxchg => (SequentiallyConsistent, Acquire),
                        _ => bx.sess().fatal("unknown ordering in atomic intrinsic"),
                    },
                    4 => match (split[2], split[3]) {
                        ("acq", "failrelaxed") if is_cxchg => (Acquire, Monotonic),
                        ("acqrel", "failrelaxed") if is_cxchg => (AcquireRelease, Monotonic),
                        _ => bx.sess().fatal("unknown ordering in atomic intrinsic"),
                    },
                    _ => bx.sess().fatal("Atomic intrinsic not in correct format"),
                };

                let invalid_monomorphization = |ty| {
                    span_invalid_monomorphization_error(
                        bx.tcx().sess,
                        span,
                        &format!(
                            "invalid monomorphization of `{}` intrinsic: \
                                  expected basic integer type, found `{}`",
                            name, ty
                        ),
                    );
                };

                match split[1] {
                    "cxchg" | "cxchgweak" => {
                        let ty = substs.type_at(0);
                        if int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_unsafe_ptr() {
                            let weak = split[1] == "cxchgweak";
                            let mut dst = args[0].immediate();
                            let mut cmp = args[1].immediate();
                            let mut src = args[2].immediate();
                            if ty.is_unsafe_ptr() {
                                // Some platforms do not support atomic operations on pointers,
                                // so we cast to integer first.
                                let ptr_llty = bx.type_ptr_to(bx.type_isize());
                                dst = bx.pointercast(dst, ptr_llty);
                                cmp = bx.ptrtoint(cmp, bx.type_isize());
                                src = bx.ptrtoint(src, bx.type_isize());
                            }
                            let pair = bx.atomic_cmpxchg(dst, cmp, src, order, failorder, weak);
                            let val = bx.extract_value(pair, 0);
                            let success = bx.extract_value(pair, 1);
                            let val = bx.from_immediate(val);
                            let success = bx.from_immediate(success);

                            let dest = result.project_field(bx, 0);
                            bx.store(val, dest.llval, dest.align);
                            let dest = result.project_field(bx, 1);
                            bx.store(success, dest.llval, dest.align);
                            return;
                        } else {
                            return invalid_monomorphization(ty);
                        }
                    }

                    "load" => {
                        let ty = substs.type_at(0);
                        if int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_unsafe_ptr() {
                            let layout = bx.layout_of(ty);
                            let size = layout.size;
                            let mut source = args[0].immediate();
                            if ty.is_unsafe_ptr() {
                                // Some platforms do not support atomic operations on pointers,
                                // so we cast to integer first...
                                let ptr_llty = bx.type_ptr_to(bx.type_isize());
                                source = bx.pointercast(source, ptr_llty);
                            }
                            let result = bx.atomic_load(source, order, size);
                            if ty.is_unsafe_ptr() {
                                // ... and then cast the result back to a pointer
                                bx.inttoptr(result, bx.backend_type(layout))
                            } else {
                                result
                            }
                        } else {
                            return invalid_monomorphization(ty);
                        }
                    }

                    "store" => {
                        let ty = substs.type_at(0);
                        if int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_unsafe_ptr() {
                            let size = bx.layout_of(ty).size;
                            let mut val = args[1].immediate();
                            let mut ptr = args[0].immediate();
                            if ty.is_unsafe_ptr() {
                                // Some platforms do not support atomic operations on pointers,
                                // so we cast to integer first.
                                let ptr_llty = bx.type_ptr_to(bx.type_isize());
                                ptr = bx.pointercast(ptr, ptr_llty);
                                val = bx.ptrtoint(val, bx.type_isize());
                            }
                            bx.atomic_store(val, ptr, order, size);
                            return;
                        } else {
                            return invalid_monomorphization(ty);
                        }
                    }

                    "fence" => {
                        bx.atomic_fence(order, SynchronizationScope::CrossThread);
                        return;
                    }

                    "singlethreadfence" => {
                        bx.atomic_fence(order, SynchronizationScope::SingleThread);
                        return;
                    }

                    // These are all AtomicRMW ops
                    op => {
                        let atom_op = match op {
                            "xchg" => AtomicRmwBinOp::AtomicXchg,
                            "xadd" => AtomicRmwBinOp::AtomicAdd,
                            "xsub" => AtomicRmwBinOp::AtomicSub,
                            "and" => AtomicRmwBinOp::AtomicAnd,
                            "nand" => AtomicRmwBinOp::AtomicNand,
                            "or" => AtomicRmwBinOp::AtomicOr,
                            "xor" => AtomicRmwBinOp::AtomicXor,
                            "max" => AtomicRmwBinOp::AtomicMax,
                            "min" => AtomicRmwBinOp::AtomicMin,
                            "umax" => AtomicRmwBinOp::AtomicUMax,
                            "umin" => AtomicRmwBinOp::AtomicUMin,
                            _ => bx.sess().fatal("unknown atomic operation"),
                        };

                        let ty = substs.type_at(0);
                        if int_type_width_signed(ty, bx.tcx()).is_some() {
                            bx.atomic_rmw(atom_op, args[0].immediate(), args[1].immediate(), order)
                        } else {
                            return invalid_monomorphization(ty);
                        }
                    }
                }
            }

            sym::nontemporal_store => {
                let dst = args[0].deref(bx.cx());
                args[1].val.nontemporal_store(bx, dst);
                return;
            }

            sym::ptr_guaranteed_eq | sym::ptr_guaranteed_ne => {
                let a = args[0].immediate();
                let b = args[1].immediate();
                if name == sym::ptr_guaranteed_eq {
                    bx.icmp(IntPredicate::IntEQ, a, b)
                } else {
                    bx.icmp(IntPredicate::IntNE, a, b)
                }
            }

            sym::ptr_offset_from => {
                let ty = substs.type_at(0);
                let pointee_size = bx.layout_of(ty).size;

                // This is the same sequence that Clang emits for pointer subtraction.
                // It can be neither `nsw` nor `nuw` because the input is treated as
                // unsigned but then the output is treated as signed, so neither works.
                let a = args[0].immediate();
                let b = args[1].immediate();
                let a = bx.ptrtoint(a, bx.type_isize());
                let b = bx.ptrtoint(b, bx.type_isize());
                let d = bx.sub(a, b);
                let pointee_size = bx.const_usize(pointee_size.bytes());
                // this is where the signed magic happens (notice the `s` in `exactsdiv`)
                bx.exactsdiv(d, pointee_size)
            }

            _ => {
                // Need to use backend-specific things in the implementation.
                bx.codegen_intrinsic_call(instance, fn_abi, args, llresult, span);
                return;
            }
        };

        if !fn_abi.ret.is_ignore() {
            if let PassMode::Cast(ty) = fn_abi.ret.mode {
                let ptr_llty = bx.type_ptr_to(bx.cast_backend_type(&ty));
                let ptr = bx.pointercast(result.llval, ptr_llty);
                bx.store(llval, ptr, result.align);
            } else {
                OperandRef::from_immediate_or_packed_pair(bx, llval, result.layout)
                    .val
                    .store(bx, result);
            }
        }
    }
}

// Returns the width of an int Ty, and if it's signed or not
// Returns None if the type is not an integer
// FIXME: there’s multiple of this functions, investigate using some of the already existing
// stuffs.
fn int_type_width_signed(ty: Ty<'_>, tcx: TyCtxt<'_>) -> Option<(u64, bool)> {
    match ty.kind() {
        ty::Int(t) => {
            Some((t.bit_width().unwrap_or(u64::from(tcx.sess.target.pointer_width)), true))
        }
        ty::Uint(t) => {
            Some((t.bit_width().unwrap_or(u64::from(tcx.sess.target.pointer_width)), false))
        }
        _ => None,
    }
}

// Returns the width of a float Ty
// Returns None if the type is not a float
fn float_type_width(ty: Ty<'_>) -> Option<u64> {
    match ty.kind() {
        ty::Float(t) => Some(t.bit_width()),
        _ => None,
    }
}
