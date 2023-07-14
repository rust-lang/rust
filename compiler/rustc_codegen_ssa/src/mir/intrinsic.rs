use super::operand::{OperandRef, OperandValue};
use super::place::PlaceRef;
use super::FunctionCx;
use crate::common::IntPredicate;
use crate::errors;
use crate::errors::InvalidMonomorphization;
use crate::glue;
use crate::meth;
use crate::traits::*;
use crate::MemFlags;

use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::{sym, Span};
use rustc_target::abi::{
    call::{FnAbi, PassMode},
    WrappingRange,
};

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

        let ty::FnDef(def_id, fn_args) = *callee_ty.kind() else {
            bug!("expected fn item type, found {}", callee_ty);
        };

        let sig = callee_ty.fn_sig(bx.tcx());
        let sig = bx.tcx().normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), sig);
        let arg_tys = sig.inputs();
        let ret_ty = sig.output();
        let name = bx.tcx().item_name(def_id);
        let name_str = name.as_str();

        let llret_ty = bx.backend_type(bx.layout_of(ret_ty));
        let result = PlaceRef::new_sized(llresult, fn_abi.ret.layout);

        let llval = match name {
            sym::abort => {
                bx.abort();
                return;
            }

            sym::va_start => bx.va_start(args[0].immediate()),
            sym::va_end => bx.va_end(args[0].immediate()),
            sym::size_of_val => {
                let tp_ty = fn_args.type_at(0);
                if let OperandValue::Pair(_, meta) = args[0].val {
                    let (llsize, _) = glue::size_and_align_of_dst(bx, tp_ty, Some(meta));
                    llsize
                } else {
                    bx.const_usize(bx.layout_of(tp_ty).size.bytes())
                }
            }
            sym::min_align_of_val => {
                let tp_ty = fn_args.type_at(0);
                if let OperandValue::Pair(_, meta) = args[0].val {
                    let (_, llalign) = glue::size_and_align_of_dst(bx, tp_ty, Some(meta));
                    llalign
                } else {
                    bx.const_usize(bx.layout_of(tp_ty).align.abi.bytes())
                }
            }
            sym::vtable_size | sym::vtable_align => {
                let vtable = args[0].immediate();
                let idx = match name {
                    sym::vtable_size => ty::COMMON_VTABLE_ENTRIES_SIZE,
                    sym::vtable_align => ty::COMMON_VTABLE_ENTRIES_ALIGN,
                    _ => bug!(),
                };
                let value = meth::VirtualIndex::from_index(idx).get_usize(bx, vtable);
                match name {
                    // Size is always <= isize::MAX.
                    sym::vtable_size => {
                        let size_bound = bx.data_layout().ptr_sized_integer().signed_max() as u128;
                        bx.range_metadata(value, WrappingRange { start: 0, end: size_bound });
                    },
                    // Alignment is always nonzero.
                    sym::vtable_align => bx.range_metadata(value, WrappingRange { start: 1, end: !0 }),
                    _ => {}
                }
                value
            }
            sym::pref_align_of
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
            sym::arith_offset => {
                let ty = fn_args.type_at(0);
                let layout = bx.layout_of(ty);
                let ptr = args[0].immediate();
                let offset = args[1].immediate();
                bx.gep(bx.backend_type(layout), ptr, &[offset])
            }
            sym::copy => {
                copy_intrinsic(
                    bx,
                    true,
                    false,
                    fn_args.type_at(0),
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
                    fn_args.type_at(0),
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
                    fn_args.type_at(0),
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
                    fn_args.type_at(0),
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
                    fn_args.type_at(0),
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
            sym::exact_div => {
                let ty = arg_tys[0];
                match int_type_width_signed(ty, bx.tcx()) {
                    Some((_width, signed)) => {
                        if signed {
                            bx.exactsdiv(args[0].immediate(), args[1].immediate())
                        } else {
                            bx.exactudiv(args[0].immediate(), args[1].immediate())
                        }
                    },
                    None => {
                        bx.tcx().sess.emit_err(InvalidMonomorphization::BasicIntegerType { span, name, ty });
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
                        bx.tcx().sess.emit_err(InvalidMonomorphization::BasicFloatType { span, name, ty: arg_tys[0] });
                        return;
                    }
                }
            }

            sym::float_to_int_unchecked => {
                if float_type_width(arg_tys[0]).is_none() {
                    bx.tcx().sess.emit_err(InvalidMonomorphization::FloatToIntUnchecked { span, ty: arg_tys[0] });
                    return;
                }
                let Some((_width, signed)) = int_type_width_signed(ret_ty, bx.tcx()) else {
                    bx.tcx().sess.emit_err(InvalidMonomorphization::FloatToIntUnchecked { span, ty: ret_ty });
                    return;
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

            sym::const_allocate => {
                // returns a null pointer at runtime.
                bx.const_null(bx.type_i8p())
            }

            sym::const_deallocate => {
                // nop at runtime.
                return;
            }

            // This requires that atomic intrinsics follow a specific naming pattern:
            // "atomic_<operation>[_<ordering>]"
            name if let Some(atomic) = name_str.strip_prefix("atomic_") => {
                use crate::common::AtomicOrdering::*;
                use crate::common::{AtomicRmwBinOp, SynchronizationScope};

                let Some((instruction, ordering)) = atomic.split_once('_') else {
                    bx.sess().emit_fatal(errors::MissingMemoryOrdering);
                };

                let parse_ordering = |bx: &Bx, s| match s {
                    "unordered" => Unordered,
                    "relaxed" => Relaxed,
                    "acquire" => Acquire,
                    "release" => Release,
                    "acqrel" => AcquireRelease,
                    "seqcst" => SequentiallyConsistent,
                    _ => bx.sess().emit_fatal(errors::UnknownAtomicOrdering),
                };

                let invalid_monomorphization = |ty| {
                    bx.tcx().sess.emit_err(InvalidMonomorphization::BasicIntegerType { span, name, ty });
                };

                match instruction {
                    "cxchg" | "cxchgweak" => {
                        let Some((success, failure)) = ordering.split_once('_') else {
                            bx.sess().emit_fatal(errors::AtomicCompareExchange);
                        };
                        let ty = fn_args.type_at(0);
                        if int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_unsafe_ptr() {
                            let weak = instruction == "cxchgweak";
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
                            let pair = bx.atomic_cmpxchg(dst, cmp, src, parse_ordering(bx, success), parse_ordering(bx, failure), weak);
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
                        let ty = fn_args.type_at(0);
                        if int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_unsafe_ptr() {
                            let layout = bx.layout_of(ty);
                            let size = layout.size;
                            let mut source = args[0].immediate();
                            if ty.is_unsafe_ptr() {
                                // Some platforms do not support atomic operations on pointers,
                                // so we cast to integer first...
                                let llty = bx.type_isize();
                                let ptr_llty = bx.type_ptr_to(llty);
                                source = bx.pointercast(source, ptr_llty);
                                let result = bx.atomic_load(llty, source, parse_ordering(bx, ordering), size);
                                // ... and then cast the result back to a pointer
                                bx.inttoptr(result, bx.backend_type(layout))
                            } else {
                                bx.atomic_load(bx.backend_type(layout), source, parse_ordering(bx, ordering), size)
                            }
                        } else {
                            return invalid_monomorphization(ty);
                        }
                    }

                    "store" => {
                        let ty = fn_args.type_at(0);
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
                            bx.atomic_store(val, ptr, parse_ordering(bx, ordering), size);
                            return;
                        } else {
                            return invalid_monomorphization(ty);
                        }
                    }

                    "fence" => {
                        bx.atomic_fence(parse_ordering(bx, ordering), SynchronizationScope::CrossThread);
                        return;
                    }

                    "singlethreadfence" => {
                        bx.atomic_fence(parse_ordering(bx, ordering), SynchronizationScope::SingleThread);
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
                            _ => bx.sess().emit_fatal(errors::UnknownAtomicOperation),
                        };

                        let ty = fn_args.type_at(0);
                        if int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_unsafe_ptr() {
                            let mut ptr = args[0].immediate();
                            let mut val = args[1].immediate();
                            if ty.is_unsafe_ptr() {
                                // Some platforms do not support atomic operations on pointers,
                                // so we cast to integer first.
                                let ptr_llty = bx.type_ptr_to(bx.type_isize());
                                ptr = bx.pointercast(ptr, ptr_llty);
                                val = bx.ptrtoint(val, bx.type_isize());
                            }
                            bx.atomic_rmw(atom_op, ptr, val, parse_ordering(bx, ordering))
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

            sym::ptr_guaranteed_cmp => {
                let a = args[0].immediate();
                let b = args[1].immediate();
                bx.icmp(IntPredicate::IntEQ, a, b)
            }

            sym::ptr_offset_from | sym::ptr_offset_from_unsigned => {
                let ty = fn_args.type_at(0);
                let pointee_size = bx.layout_of(ty).size;

                let a = args[0].immediate();
                let b = args[1].immediate();
                let a = bx.ptrtoint(a, bx.type_isize());
                let b = bx.ptrtoint(b, bx.type_isize());
                let pointee_size = bx.const_usize(pointee_size.bytes());
                if name == sym::ptr_offset_from {
                    // This is the same sequence that Clang emits for pointer subtraction.
                    // It can be neither `nsw` nor `nuw` because the input is treated as
                    // unsigned but then the output is treated as signed, so neither works.
                    let d = bx.sub(a, b);
                    // this is where the signed magic happens (notice the `s` in `exactsdiv`)
                    bx.exactsdiv(d, pointee_size)
                } else {
                    // The `_unsigned` version knows the relative ordering of the pointers,
                    // so can use `sub nuw` and `udiv exact` instead of dealing in signed.
                    let d = bx.unchecked_usub(a, b);
                    bx.exactudiv(d, pointee_size)
                }
            }

            _ => {
                // Need to use backend-specific things in the implementation.
                bx.codegen_intrinsic_call(instance, fn_abi, args, llresult, span);
                return;
            }
        };

        if !fn_abi.ret.is_ignore() {
            if let PassMode::Cast(ty, _) = &fn_abi.ret.mode {
                let ptr_llty = bx.type_ptr_to(bx.cast_backend_type(ty));
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
