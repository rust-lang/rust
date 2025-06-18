use rustc_abi::WrappingRange;
use rustc_middle::bug;
use rustc_middle::mir::SourceInfo;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::config::OptLevel;
use rustc_span::sym;

use super::FunctionCx;
use super::operand::OperandRef;
use super::place::PlaceRef;
use crate::common::{AtomicRmwBinOp, SynchronizationScope};
use crate::errors::InvalidMonomorphization;
use crate::traits::*;
use crate::{MemFlags, meth, size_of_val};

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
    /// In the `Err` case, returns the instance that should be called instead.
    pub fn codegen_intrinsic_call(
        &mut self,
        bx: &mut Bx,
        instance: ty::Instance<'tcx>,
        args: &[OperandRef<'tcx, Bx::Value>],
        result: PlaceRef<'tcx, Bx::Value>,
        source_info: SourceInfo,
    ) -> Result<(), ty::Instance<'tcx>> {
        let span = source_info.span;

        let name = bx.tcx().item_name(instance.def_id());
        let fn_args = instance.args;

        // If we're swapping something that's *not* an `OperandValue::Ref`,
        // then we can do it directly and avoid the alloca.
        // Otherwise, we'll let the fallback MIR body take care of it.
        if let sym::typed_swap_nonoverlapping = name {
            let pointee_ty = fn_args.type_at(0);
            let pointee_layout = bx.layout_of(pointee_ty);
            if !bx.is_backend_ref(pointee_layout)
                // But if we're not going to optimize, trying to use the fallback
                // body just makes things worse, so don't bother.
                || bx.sess().opts.optimize == OptLevel::No
                // NOTE(eddyb) SPIR-V's Logical addressing model doesn't allow for arbitrary
                // reinterpretation of values as (chunkable) byte arrays, and the loop in the
                // block optimization in `ptr::swap_nonoverlapping` is hard to rewrite back
                // into the (unoptimized) direct swapping implementation, so we disable it.
                || bx.sess().target.arch == "spirv"
            {
                let align = pointee_layout.align.abi;
                let x_place = args[0].val.deref(align);
                let y_place = args[1].val.deref(align);
                bx.typed_place_swap(x_place, y_place, pointee_layout);
                return Ok(());
            }
        }

        let invalid_monomorphization_int_type = |ty| {
            bx.tcx().dcx().emit_err(InvalidMonomorphization::BasicIntegerType { span, name, ty });
        };

        let parse_atomic_ordering = |ord: ty::Value<'tcx>| {
            let discr = ord.valtree.unwrap_branch()[0].unwrap_leaf();
            discr.to_atomic_ordering()
        };

        let llval = match name {
            sym::abort => {
                bx.abort();
                return Ok(());
            }

            sym::caller_location => {
                let location = self.get_caller_location(bx, source_info);
                location.val.store(bx, result);
                return Ok(());
            }

            sym::va_start => bx.va_start(args[0].immediate()),
            sym::va_end => bx.va_end(args[0].immediate()),
            sym::size_of_val => {
                let tp_ty = fn_args.type_at(0);
                let (_, meta) = args[0].val.pointer_parts();
                let (llsize, _) = size_of_val::size_and_align_of_dst(bx, tp_ty, meta);
                llsize
            }
            sym::align_of_val => {
                let tp_ty = fn_args.type_at(0);
                let (_, meta) = args[0].val.pointer_parts();
                let (_, llalign) = size_of_val::size_and_align_of_dst(bx, tp_ty, meta);
                llalign
            }
            sym::vtable_size | sym::vtable_align => {
                let vtable = args[0].immediate();
                let idx = match name {
                    sym::vtable_size => ty::COMMON_VTABLE_ENTRIES_SIZE,
                    sym::vtable_align => ty::COMMON_VTABLE_ENTRIES_ALIGN,
                    _ => bug!(),
                };
                let value = meth::VirtualIndex::from_index(idx).get_usize(
                    bx,
                    vtable,
                    instance.ty(bx.tcx(), bx.typing_env()),
                );
                match name {
                    // Size is always <= isize::MAX.
                    sym::vtable_size => {
                        let size_bound = bx.data_layout().ptr_sized_integer().signed_max() as u128;
                        bx.range_metadata(value, WrappingRange { start: 0, end: size_bound });
                    }
                    // Alignment is always nonzero.
                    sym::vtable_align => {
                        bx.range_metadata(value, WrappingRange { start: 1, end: !0 })
                    }
                    _ => {}
                }
                value
            }
            sym::needs_drop | sym::type_id | sym::type_name | sym::variant_count => {
                let value = bx.tcx().const_eval_instance(bx.typing_env(), instance, span).unwrap();
                OperandRef::from_const(bx, value, result.layout.ty).immediate_or_packed_pair(bx)
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
                return Ok(());
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
                return Ok(());
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
                return Ok(());
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
                return Ok(());
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
                return Ok(());
            }
            sym::volatile_store => {
                let dst = args[0].deref(bx.cx());
                args[1].val.volatile_store(bx, dst);
                return Ok(());
            }
            sym::unaligned_volatile_store => {
                let dst = args[0].deref(bx.cx());
                args[1].val.unaligned_volatile_store(bx, dst);
                return Ok(());
            }
            sym::disjoint_bitor => {
                let a = args[0].immediate();
                let b = args[1].immediate();
                bx.or_disjoint(a, b)
            }
            sym::exact_div => {
                let ty = args[0].layout.ty;
                match int_type_width_signed(ty, bx.tcx()) {
                    Some((_width, signed)) => {
                        if signed {
                            bx.exactsdiv(args[0].immediate(), args[1].immediate())
                        } else {
                            bx.exactudiv(args[0].immediate(), args[1].immediate())
                        }
                    }
                    None => {
                        bx.tcx().dcx().emit_err(InvalidMonomorphization::BasicIntegerType {
                            span,
                            name,
                            ty,
                        });
                        return Ok(());
                    }
                }
            }
            sym::fadd_fast | sym::fsub_fast | sym::fmul_fast | sym::fdiv_fast | sym::frem_fast => {
                match float_type_width(args[0].layout.ty) {
                    Some(_width) => match name {
                        sym::fadd_fast => bx.fadd_fast(args[0].immediate(), args[1].immediate()),
                        sym::fsub_fast => bx.fsub_fast(args[0].immediate(), args[1].immediate()),
                        sym::fmul_fast => bx.fmul_fast(args[0].immediate(), args[1].immediate()),
                        sym::fdiv_fast => bx.fdiv_fast(args[0].immediate(), args[1].immediate()),
                        sym::frem_fast => bx.frem_fast(args[0].immediate(), args[1].immediate()),
                        _ => bug!(),
                    },
                    None => {
                        bx.tcx().dcx().emit_err(InvalidMonomorphization::BasicFloatType {
                            span,
                            name,
                            ty: args[0].layout.ty,
                        });
                        return Ok(());
                    }
                }
            }
            sym::fadd_algebraic
            | sym::fsub_algebraic
            | sym::fmul_algebraic
            | sym::fdiv_algebraic
            | sym::frem_algebraic => match float_type_width(args[0].layout.ty) {
                Some(_width) => match name {
                    sym::fadd_algebraic => {
                        bx.fadd_algebraic(args[0].immediate(), args[1].immediate())
                    }
                    sym::fsub_algebraic => {
                        bx.fsub_algebraic(args[0].immediate(), args[1].immediate())
                    }
                    sym::fmul_algebraic => {
                        bx.fmul_algebraic(args[0].immediate(), args[1].immediate())
                    }
                    sym::fdiv_algebraic => {
                        bx.fdiv_algebraic(args[0].immediate(), args[1].immediate())
                    }
                    sym::frem_algebraic => {
                        bx.frem_algebraic(args[0].immediate(), args[1].immediate())
                    }
                    _ => bug!(),
                },
                None => {
                    bx.tcx().dcx().emit_err(InvalidMonomorphization::BasicFloatType {
                        span,
                        name,
                        ty: args[0].layout.ty,
                    });
                    return Ok(());
                }
            },

            sym::float_to_int_unchecked => {
                if float_type_width(args[0].layout.ty).is_none() {
                    bx.tcx().dcx().emit_err(InvalidMonomorphization::FloatToIntUnchecked {
                        span,
                        ty: args[0].layout.ty,
                    });
                    return Ok(());
                }
                let Some((_width, signed)) = int_type_width_signed(result.layout.ty, bx.tcx())
                else {
                    bx.tcx().dcx().emit_err(InvalidMonomorphization::FloatToIntUnchecked {
                        span,
                        ty: result.layout.ty,
                    });
                    return Ok(());
                };
                if signed {
                    bx.fptosi(args[0].immediate(), bx.backend_type(result.layout))
                } else {
                    bx.fptoui(args[0].immediate(), bx.backend_type(result.layout))
                }
            }

            sym::atomic_load => {
                let ty = fn_args.type_at(0);
                if !(int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_raw_ptr()) {
                    invalid_monomorphization_int_type(ty);
                    return Ok(());
                }
                let ordering = fn_args.const_at(1).to_value();
                let layout = bx.layout_of(ty);
                let source = args[0].immediate();
                bx.atomic_load(
                    bx.backend_type(layout),
                    source,
                    parse_atomic_ordering(ordering),
                    layout.size,
                )
            }
            sym::atomic_store => {
                let ty = fn_args.type_at(0);
                if !(int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_raw_ptr()) {
                    invalid_monomorphization_int_type(ty);
                    return Ok(());
                }
                let ordering = fn_args.const_at(1).to_value();
                let size = bx.layout_of(ty).size;
                let val = args[1].immediate();
                let ptr = args[0].immediate();
                bx.atomic_store(val, ptr, parse_atomic_ordering(ordering), size);
                return Ok(());
            }
            sym::atomic_cxchg | sym::atomic_cxchgweak => {
                let ty = fn_args.type_at(0);
                if !(int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_raw_ptr()) {
                    invalid_monomorphization_int_type(ty);
                    return Ok(());
                }
                let succ_ordering = fn_args.const_at(1).to_value();
                let fail_ordering = fn_args.const_at(2).to_value();
                let weak = name == sym::atomic_cxchgweak;
                let dst = args[0].immediate();
                let cmp = args[1].immediate();
                let src = args[2].immediate();
                let (val, success) = bx.atomic_cmpxchg(
                    dst,
                    cmp,
                    src,
                    parse_atomic_ordering(succ_ordering),
                    parse_atomic_ordering(fail_ordering),
                    weak,
                );
                let val = bx.from_immediate(val);
                let success = bx.from_immediate(success);

                let dest = result.project_field(bx, 0);
                bx.store_to_place(val, dest.val);
                let dest = result.project_field(bx, 1);
                bx.store_to_place(success, dest.val);

                return Ok(());
            }
            // These are all AtomicRMW ops
            sym::atomic_max | sym::atomic_min => {
                let atom_op = if name == sym::atomic_max {
                    AtomicRmwBinOp::AtomicMax
                } else {
                    AtomicRmwBinOp::AtomicMin
                };

                let ty = fn_args.type_at(0);
                if matches!(ty.kind(), ty::Int(_)) {
                    let ordering = fn_args.const_at(1).to_value();
                    let ptr = args[0].immediate();
                    let val = args[1].immediate();
                    bx.atomic_rmw(atom_op, ptr, val, parse_atomic_ordering(ordering))
                } else {
                    invalid_monomorphization_int_type(ty);
                    return Ok(());
                }
            }
            sym::atomic_umax | sym::atomic_umin => {
                let atom_op = if name == sym::atomic_umax {
                    AtomicRmwBinOp::AtomicUMax
                } else {
                    AtomicRmwBinOp::AtomicUMin
                };

                let ty = fn_args.type_at(0);
                if matches!(ty.kind(), ty::Uint(_)) {
                    let ordering = fn_args.const_at(1).to_value();
                    let ptr = args[0].immediate();
                    let val = args[1].immediate();
                    bx.atomic_rmw(atom_op, ptr, val, parse_atomic_ordering(ordering))
                } else {
                    invalid_monomorphization_int_type(ty);
                    return Ok(());
                }
            }
            sym::atomic_xchg
            | sym::atomic_xadd
            | sym::atomic_xsub
            | sym::atomic_and
            | sym::atomic_nand
            | sym::atomic_or
            | sym::atomic_xor => {
                let atom_op = match name {
                    sym::atomic_xchg => AtomicRmwBinOp::AtomicXchg,
                    sym::atomic_xadd => AtomicRmwBinOp::AtomicAdd,
                    sym::atomic_xsub => AtomicRmwBinOp::AtomicSub,
                    sym::atomic_and => AtomicRmwBinOp::AtomicAnd,
                    sym::atomic_nand => AtomicRmwBinOp::AtomicNand,
                    sym::atomic_or => AtomicRmwBinOp::AtomicOr,
                    sym::atomic_xor => AtomicRmwBinOp::AtomicXor,
                    _ => unreachable!(),
                };

                let ty = fn_args.type_at(0);
                if int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_raw_ptr() {
                    let ordering = fn_args.const_at(1).to_value();
                    let ptr = args[0].immediate();
                    let val = args[1].immediate();
                    bx.atomic_rmw(atom_op, ptr, val, parse_atomic_ordering(ordering))
                } else {
                    invalid_monomorphization_int_type(ty);
                    return Ok(());
                }
            }
            sym::atomic_fence => {
                let ordering = fn_args.const_at(0).to_value();
                bx.atomic_fence(parse_atomic_ordering(ordering), SynchronizationScope::CrossThread);
                return Ok(());
            }

            sym::atomic_singlethreadfence => {
                let ordering = fn_args.const_at(0).to_value();
                bx.atomic_fence(
                    parse_atomic_ordering(ordering),
                    SynchronizationScope::SingleThread,
                );
                return Ok(());
            }

            sym::nontemporal_store => {
                let dst = args[0].deref(bx.cx());
                args[1].val.nontemporal_store(bx, dst);
                return Ok(());
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

            sym::cold_path => {
                // This is a no-op. The intrinsic is just a hint to the optimizer.
                return Ok(());
            }

            _ => {
                // Need to use backend-specific things in the implementation.
                return bx.codegen_intrinsic_call(instance, args, result, span);
            }
        };

        if result.layout.ty.is_bool() {
            let val = bx.from_immediate(llval);
            bx.store_to_place(val, result.val);
        } else if !result.layout.ty.is_unit() {
            bx.store_to_place(llval, result.val);
        }
        Ok(())
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
