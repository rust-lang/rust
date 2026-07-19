use rustc_abi::{Align, FieldIdx, WrappingRange};
use rustc_middle::mir::SourceInfo;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_session::config::OptLevel;
use rustc_span::{ErrorGuaranteed, sym};
use rustc_target::spec::Arch;

use super::operand::{OperandRef, OperandValue};
use super::place::PlaceValue;
use super::{FunctionCx, IntrinsicResult};
use crate::common::{AtomicRmwBinOp, SynchronizationScope};
use crate::diagnostics::InvalidMonomorphization;
use crate::mir::operand::OperandRefBuilder;
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
    let size = bx.unchecked_sumul(bx.const_usize(size.bytes()), count);
    let flags = if volatile { MemFlags::VOLATILE } else { MemFlags::empty() };
    if allow_overlap {
        bx.memmove(dst, align, src, align, size, flags);
    } else {
        bx.memcpy(dst, align, src, align, size, flags, None);
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
    /// In the `Fallback` case, returns the instance that should be called instead.
    pub fn codegen_intrinsic_call(
        &mut self,
        bx: &mut Bx,
        instance: ty::Instance<'tcx>,
        args: &[OperandRef<'tcx, Bx::Value>],
        result_layout: ty::layout::TyAndLayout<'tcx>,
        result_place: Option<PlaceValue<Bx::Value>>,
        source_info: SourceInfo,
    ) -> IntrinsicResult<'tcx, Bx::Value> {
        // When `-Zforce-intrinsic-fallback` is enabled, always use the fallback body if it exists,
        if bx.tcx().sess.opts.unstable_opts.force_intrinsic_fallback
            && let Some(def) = bx.tcx().intrinsic(instance.def_id())
            && !def.must_be_overridden
        {
            return IntrinsicResult::Fallback(ty::Instance::new_raw(
                instance.def_id(),
                instance.args,
            ));
        }

        let span = source_info.span;

        let name = bx.tcx().item_name(instance.def_id());
        let fn_args = instance.args;

        // If we're swapping something that's *not* an `OperandValue::Ref`,
        // then we can do it directly and avoid the alloca.
        // Otherwise, we'll let the fallback MIR body take care of it.
        if let sym::typed_swap_nonoverlapping = name {
            let pointee_ty = fn_args.type_at(0);
            let pointee_layout = bx.layout_of(pointee_ty);
            if pointee_layout.is_ssa_standalone()
                // But if we're not going to optimize, trying to use the fallback
                // body just makes things worse, so don't bother.
                || bx.sess().opts.optimize == OptLevel::No
                // NOTE(eddyb) SPIR-V's Logical addressing model doesn't allow for arbitrary
                // reinterpretation of values as (chunkable) byte arrays, and the loop in the
                // block optimization in `ptr::swap_nonoverlapping` is hard to rewrite back
                // into the (unoptimized) direct swapping implementation, so we disable it.
                || bx.sess().target.arch == Arch::SpirV
            {
                let align = pointee_layout.align.abi;
                let x_place = args[0].val.deref(align);
                let y_place = args[1].val.deref(align);
                bx.typed_place_swap(x_place, y_place, pointee_layout);
                return IntrinsicResult::Operand(OperandValue::ZeroSized);
            }
        }

        let invalid_monomorphization_int_type = |ty| -> ErrorGuaranteed {
            bx.tcx().dcx().emit_err(InvalidMonomorphization::BasicIntegerType { span, name, ty })
        };
        let invalid_monomorphization_int_or_ptr_type = |ty| -> ErrorGuaranteed {
            bx.tcx().dcx().emit_err(InvalidMonomorphization::BasicIntegerOrPtrType {
                span,
                name,
                ty,
            })
        };

        let parse_atomic_ordering = |ord: ty::Value<'tcx>| {
            let discr = ord.to_branch()[0].to_leaf();
            discr.to_atomic_ordering()
        };

        if args.is_empty() {
            match name {
                sym::abort
                | sym::unreachable
                | sym::cold_path
                | sym::gpu_launch_sized_workgroup_mem
                | sym::breakpoint
                | sym::amdgpu_dispatch_ptr
                | sym::assert_zero_valid
                | sym::assert_mem_uninitialized_valid
                | sym::assert_inhabited
                | sym::ub_checks
                | sym::contract_checks
                | sym::atomic_fence
                | sym::atomic_singlethreadfence
                | sym::caller_location
                | sym::return_address => {}
                _ => {
                    span_bug!(
                        span,
                        "Nullary intrinsic {name} must be called in a const block. \
                        If you are seeing this message from code outside the standard library, the \
                        unstable implementation details of the relevant intrinsic may have changed. \
                        Consider using stable APIs instead. \
                        If you are adding a new nullary intrinsic that is inherently a runtime \
                        intrinsic, update this check."
                    );
                }
            }
        }

        let op_val: OperandValue<_> = match name {
            sym::abort => {
                bx.abort();
                OperandValue::ZeroSized
            }

            sym::caller_location => {
                let location = self.get_caller_location(bx, source_info);
                location.val
            }

            // va_end uses the fallback body (a no-op).
            sym::va_start => {
                bx.va_start(args[0].immediate());
                OperandValue::ZeroSized
            }

            sym::size_of_val => {
                let tp_ty = fn_args.type_at(0);
                let (_, meta) = args[0].val.pointer_parts();
                let (llsize, _) = size_of_val::size_and_align_of_dst(bx, tp_ty, meta);
                OperandValue::Immediate(llsize)
            }
            sym::align_of_val => {
                let tp_ty = fn_args.type_at(0);
                let (_, meta) = args[0].val.pointer_parts();
                let (_, llalign) = size_of_val::size_and_align_of_dst(bx, tp_ty, meta);
                OperandValue::Immediate(llalign)
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
                    // Alignment is always a power of two, thus 1..=0x800…000,
                    // but also bounded by the maximum we support in type layout.
                    sym::vtable_align => {
                        let align_bound = Align::max_for_target(bx.data_layout()).bytes().into();
                        bx.range_metadata(value, WrappingRange { start: 1, end: align_bound })
                    }
                    _ => {}
                }
                OperandValue::Immediate(value)
            }
            sym::arith_offset => {
                let ty = fn_args.type_at(0);
                let layout = bx.layout_of(ty);
                let ptr = args[0].immediate();
                let offset = args[1].immediate();
                OperandValue::Immediate(bx.gep(bx.backend_type(layout), ptr, &[offset]))
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
                OperandValue::ZeroSized
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
                OperandValue::ZeroSized
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
                OperandValue::ZeroSized
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
                OperandValue::ZeroSized
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
                OperandValue::ZeroSized
            }
            sym::volatile_store | sym::unaligned_volatile_store => {
                let dst = args[0].deref(bx.cx());
                let dst = if name == sym::volatile_store { dst } else { dst.unaligned() };
                args[1].val.volatile_store(bx, dst);
                OperandValue::ZeroSized
            }
            sym::disjoint_bitor => {
                let a = args[0].immediate();
                let b = args[1].immediate();
                OperandValue::Immediate(bx.or_disjoint(a, b))
            }
            sym::exact_div => {
                let ty = args[0].layout.ty;
                match int_type_width_signed(ty, bx.tcx()) {
                    Some((_width, signed)) => OperandValue::Immediate(if signed {
                        bx.exactsdiv(args[0].immediate(), args[1].immediate())
                    } else {
                        bx.exactudiv(args[0].immediate(), args[1].immediate())
                    }),
                    None => {
                        let err = bx
                            .tcx()
                            .dcx()
                            .emit_err(InvalidMonomorphization::BasicIntegerType { span, name, ty });
                        return IntrinsicResult::Err(err);
                    }
                }
            }
            sym::fadd_fast | sym::fsub_fast | sym::fmul_fast | sym::fdiv_fast | sym::frem_fast => {
                match float_type_width(args[0].layout.ty) {
                    Some(_width) => OperandValue::Immediate(match name {
                        sym::fadd_fast => bx.fadd_fast(args[0].immediate(), args[1].immediate()),
                        sym::fsub_fast => bx.fsub_fast(args[0].immediate(), args[1].immediate()),
                        sym::fmul_fast => bx.fmul_fast(args[0].immediate(), args[1].immediate()),
                        sym::fdiv_fast => bx.fdiv_fast(args[0].immediate(), args[1].immediate()),
                        sym::frem_fast => bx.frem_fast(args[0].immediate(), args[1].immediate()),
                        _ => bug!(),
                    }),
                    None => {
                        let err =
                            bx.tcx().dcx().emit_err(InvalidMonomorphization::BasicFloatType {
                                span,
                                name,
                                ty: args[0].layout.ty,
                            });
                        return IntrinsicResult::Err(err);
                    }
                }
            }
            sym::fadd_algebraic
            | sym::fsub_algebraic
            | sym::fmul_algebraic
            | sym::fdiv_algebraic
            | sym::frem_algebraic => match float_type_width(args[0].layout.ty) {
                Some(_width) => OperandValue::Immediate(match name {
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
                }),
                None => {
                    let err = bx.tcx().dcx().emit_err(InvalidMonomorphization::BasicFloatType {
                        span,
                        name,
                        ty: args[0].layout.ty,
                    });
                    return IntrinsicResult::Err(err);
                }
            },

            sym::float_to_int_unchecked => {
                if float_type_width(args[0].layout.ty).is_none() {
                    let err =
                        bx.tcx().dcx().emit_err(InvalidMonomorphization::FloatToIntUnchecked {
                            span,
                            ty: args[0].layout.ty,
                        });
                    return IntrinsicResult::Err(err);
                }
                let Some((_width, signed)) = int_type_width_signed(result_layout.ty, bx.tcx())
                else {
                    let err =
                        bx.tcx().dcx().emit_err(InvalidMonomorphization::FloatToIntUnchecked {
                            span,
                            ty: result_layout.ty,
                        });
                    return IntrinsicResult::Err(err);
                };
                OperandValue::Immediate(if signed {
                    bx.fptosi(args[0].immediate(), bx.backend_type(result_layout))
                } else {
                    bx.fptoui(args[0].immediate(), bx.backend_type(result_layout))
                })
            }

            sym::atomic_load => {
                let ty = fn_args.type_at(0);
                if !(int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_raw_ptr()) {
                    let err = invalid_monomorphization_int_or_ptr_type(ty);
                    return IntrinsicResult::Err(err);
                }
                let ordering = fn_args.const_at(1).to_value();
                let layout = bx.layout_of(ty);
                let source = args[0].immediate();
                OperandValue::Immediate(bx.atomic_load(
                    bx.backend_type(layout),
                    source,
                    parse_atomic_ordering(ordering),
                    layout.size,
                ))
            }
            sym::atomic_store => {
                let ty = fn_args.type_at(0);
                if !(int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_raw_ptr()) {
                    let err = invalid_monomorphization_int_or_ptr_type(ty);
                    return IntrinsicResult::Err(err);
                }
                let ordering = fn_args.const_at(1).to_value();
                let size = bx.layout_of(ty).size;
                let val = args[1].immediate();
                let ptr = args[0].immediate();
                bx.atomic_store(val, ptr, parse_atomic_ordering(ordering), size);
                OperandValue::ZeroSized
            }
            // These are all AtomicRMW ops
            sym::atomic_cxchg | sym::atomic_cxchgweak => {
                let ty = fn_args.type_at(0);
                if !(int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_raw_ptr()) {
                    let err = invalid_monomorphization_int_or_ptr_type(ty);
                    return IntrinsicResult::Err(err);
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

                let mut builder = OperandRefBuilder::new(result_layout);
                builder.insert_imm(FieldIdx::from_u32(0), val);
                builder.insert_imm(FieldIdx::from_u32(1), success);
                builder.build(bx.cx()).val
            }
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
                    OperandValue::Immediate(bx.atomic_rmw(
                        atom_op,
                        ptr,
                        val,
                        parse_atomic_ordering(ordering),
                        /* ret_ptr */ false,
                    ))
                } else {
                    let err = invalid_monomorphization_int_type(ty);
                    return IntrinsicResult::Err(err);
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
                    OperandValue::Immediate(bx.atomic_rmw(
                        atom_op,
                        ptr,
                        val,
                        parse_atomic_ordering(ordering),
                        /* ret_ptr */ false,
                    ))
                } else {
                    let err = invalid_monomorphization_int_type(ty);
                    return IntrinsicResult::Err(err);
                }
            }
            sym::atomic_xchg => {
                let ty = fn_args.type_at(0);
                let ordering = fn_args.const_at(1).to_value();
                if int_type_width_signed(ty, bx.tcx()).is_some() || ty.is_raw_ptr() {
                    let ptr = args[0].immediate();
                    let val = args[1].immediate();
                    let atomic_op = AtomicRmwBinOp::AtomicXchg;
                    OperandValue::Immediate(bx.atomic_rmw(
                        atomic_op,
                        ptr,
                        val,
                        parse_atomic_ordering(ordering),
                        /* ret_ptr */ ty.is_raw_ptr(),
                    ))
                } else {
                    let err = invalid_monomorphization_int_or_ptr_type(ty);
                    return IntrinsicResult::Err(err);
                }
            }
            sym::atomic_xadd
            | sym::atomic_xsub
            | sym::atomic_and
            | sym::atomic_nand
            | sym::atomic_or
            | sym::atomic_xor => {
                let atom_op = match name {
                    sym::atomic_xadd => AtomicRmwBinOp::AtomicAdd,
                    sym::atomic_xsub => AtomicRmwBinOp::AtomicSub,
                    sym::atomic_and => AtomicRmwBinOp::AtomicAnd,
                    sym::atomic_nand => AtomicRmwBinOp::AtomicNand,
                    sym::atomic_or => AtomicRmwBinOp::AtomicOr,
                    sym::atomic_xor => AtomicRmwBinOp::AtomicXor,
                    _ => unreachable!(),
                };

                // The type of the in-memory data.
                let ty_mem = fn_args.type_at(0);
                // The type of the 2nd operand, given by-value.
                let ty_op = fn_args.type_at(1);

                let ordering = fn_args.const_at(2).to_value();
                // We require either both arguments to have the same integer type, or the first to
                // be a pointer and the second to be `usize`.
                if (int_type_width_signed(ty_mem, bx.tcx()).is_some() && ty_op == ty_mem)
                    || (ty_mem.is_raw_ptr() && ty_op == bx.tcx().types.usize)
                {
                    let ptr = args[0].immediate(); // of type "pointer to `ty_mem`"
                    let val = args[1].immediate(); // of type `ty_op`
                    OperandValue::Immediate(bx.atomic_rmw(
                        atom_op,
                        ptr,
                        val,
                        parse_atomic_ordering(ordering),
                        /* ret_ptr */ ty_mem.is_raw_ptr(),
                    ))
                } else {
                    let err = invalid_monomorphization_int_or_ptr_type(ty_mem);
                    return IntrinsicResult::Err(err);
                }
            }
            sym::atomic_fence => {
                let ordering = fn_args.const_at(0).to_value();
                bx.atomic_fence(parse_atomic_ordering(ordering), SynchronizationScope::CrossThread);
                OperandValue::ZeroSized
            }

            sym::atomic_singlethreadfence => {
                let ordering = fn_args.const_at(0).to_value();
                bx.atomic_fence(
                    parse_atomic_ordering(ordering),
                    SynchronizationScope::SingleThread,
                );
                OperandValue::ZeroSized
            }

            sym::nontemporal_store => {
                let dst = args[0].deref(bx.cx());
                args[1].val.nontemporal_store(bx, dst);
                OperandValue::ZeroSized
            }

            sym::ptr_offset_from | sym::ptr_offset_from_unsigned => {
                let ty = fn_args.type_at(0);
                let pointee_size = bx.layout_of(ty).size;

                let a = args[0].immediate();
                let b = args[1].immediate();
                let a = bx.ptrtoint(a, bx.type_isize());
                let b = bx.ptrtoint(b, bx.type_isize());
                let pointee_size = bx.const_usize(pointee_size.bytes());
                OperandValue::Immediate(if name == sym::ptr_offset_from {
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
                })
            }

            sym::cold_path => {
                // This is a no-op. The intrinsic is just a hint to the optimizer.
                OperandValue::ZeroSized
            }

            _ => {
                // Need to use backend-specific things in the implementation.
                let result =
                    bx.codegen_intrinsic_call(instance, args, result_layout, result_place, span);
                if let IntrinsicResult::Operand(op) = result {
                    op
                } else {
                    return result;
                }
            }
        };

        debug_assert!(
            op_val.is_expected_variant_for_type(result_layout),
            "[{name:?}] Value {op_val:?} is wrong for type {result_layout:?}",
        );

        IntrinsicResult::Operand(op_val)
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
