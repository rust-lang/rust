//! Intrinsics and other functions that the interpreter executes without
//! looking at their MIR. Intrinsics/functions supported here are shared by CTFE
//! and miri.

use std::assert_matches::assert_matches;

use rustc_abi::Size;
use rustc_apfloat::ieee::{Double, Half, Quad, Single};
use rustc_hir::def_id::DefId;
use rustc_middle::mir::{self, BinOp, ConstValue, NonDivergingIntrinsic};
use rustc_middle::ty::layout::{LayoutOf as _, TyAndLayout, ValidityRequirement};
use rustc_middle::ty::{GenericArgsRef, Ty, TyCtxt};
use rustc_middle::{bug, ty};
use rustc_span::{Symbol, sym};
use tracing::trace;

use super::memory::MemoryKind;
use super::util::ensure_monomorphic_enough;
use super::{
    Allocation, CheckInAllocMsg, ConstAllocation, GlobalId, ImmTy, InterpCx, InterpResult, Machine,
    OpTy, PlaceTy, Pointer, PointerArithmetic, Provenance, Scalar, err_inval, err_ub_custom,
    err_unsup_format, interp_ok, throw_inval, throw_ub_custom, throw_ub_format,
};
use crate::fluent_generated as fluent;

/// Directly returns an `Allocation` containing an absolute path representation of the given type.
pub(crate) fn alloc_type_name<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> ConstAllocation<'tcx> {
    let path = crate::util::type_name(tcx, ty);
    let alloc = Allocation::from_bytes_byte_aligned_immutable(path.into_bytes());
    tcx.mk_const_alloc(alloc)
}

/// The logic for all nullary intrinsics is implemented here. These intrinsics don't get evaluated
/// inside an `InterpCx` and instead have their value computed directly from rustc internal info.
pub(crate) fn eval_nullary_intrinsic<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    def_id: DefId,
    args: GenericArgsRef<'tcx>,
) -> InterpResult<'tcx, ConstValue<'tcx>> {
    let tp_ty = args.type_at(0);
    let name = tcx.item_name(def_id);
    interp_ok(match name {
        sym::type_name => {
            ensure_monomorphic_enough(tcx, tp_ty)?;
            let alloc = alloc_type_name(tcx, tp_ty);
            ConstValue::Slice { data: alloc, meta: alloc.inner().size().bytes() }
        }
        sym::needs_drop => {
            ensure_monomorphic_enough(tcx, tp_ty)?;
            ConstValue::from_bool(tp_ty.needs_drop(tcx, typing_env))
        }
        sym::pref_align_of => {
            // Correctly handles non-monomorphic calls, so there is no need for ensure_monomorphic_enough.
            let layout = tcx
                .layout_of(typing_env.as_query_input(tp_ty))
                .map_err(|e| err_inval!(Layout(*e)))?;
            ConstValue::from_target_usize(layout.align.pref.bytes(), &tcx)
        }
        sym::type_id => {
            ensure_monomorphic_enough(tcx, tp_ty)?;
            ConstValue::from_u128(tcx.type_id_hash(tp_ty).as_u128())
        }
        sym::variant_count => match match tp_ty.kind() {
            // Pattern types have the same number of variants as their base type.
            // Even if we restrict e.g. which variants are valid, the variants are essentially just uninhabited.
            // And `Result<(), !>` still has two variants according to `variant_count`.
            ty::Pat(base, _) => *base,
            _ => tp_ty,
        }
        .kind()
        {
            // Correctly handles non-monomorphic calls, so there is no need for ensure_monomorphic_enough.
            ty::Adt(adt, _) => ConstValue::from_target_usize(adt.variants().len() as u64, &tcx),
            ty::Alias(..) | ty::Param(_) | ty::Placeholder(_) | ty::Infer(_) => {
                throw_inval!(TooGeneric)
            }
            ty::Pat(..) => unreachable!(),
            ty::Bound(_, _) => bug!("bound ty during ctfe"),
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Foreign(_)
            | ty::Str
            | ty::Array(_, _)
            | ty::Slice(_)
            | ty::RawPtr(_, _)
            | ty::Ref(_, _, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(..)
            | ty::Dynamic(_, _, _)
            | ty::Closure(_, _)
            | ty::CoroutineClosure(_, _)
            | ty::Coroutine(_, _)
            | ty::CoroutineWitness(..)
            | ty::UnsafeBinder(_)
            | ty::Never
            | ty::Tuple(_)
            | ty::Error(_) => ConstValue::from_target_usize(0u64, &tcx),
        },
        other => bug!("`{}` is not a zero arg intrinsic", other),
    })
}

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    /// Returns `true` if emulation happened.
    /// Here we implement the intrinsics that are common to all Miri instances; individual machines can add their own
    /// intrinsic handling.
    pub fn eval_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, M::Provenance>],
        dest: &PlaceTy<'tcx, M::Provenance>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, bool> {
        let instance_args = instance.args;
        let intrinsic_name = self.tcx.item_name(instance.def_id());

        match intrinsic_name {
            sym::caller_location => {
                let span = self.find_closest_untracked_caller_location();
                let val = self.tcx.span_as_caller_location(span);
                let val =
                    self.const_val_to_op(val, self.tcx.caller_location_ty(), Some(dest.layout))?;
                self.copy_op(&val, dest)?;
            }

            sym::min_align_of_val | sym::size_of_val => {
                // Avoid `deref_pointer` -- this is not a deref, the ptr does not have to be
                // dereferenceable!
                let place = self.ref_to_mplace(&self.read_immediate(&args[0])?)?;
                let (size, align) = self
                    .size_and_align_of_mplace(&place)?
                    .ok_or_else(|| err_unsup_format!("`extern type` does not have known layout"))?;

                let result = match intrinsic_name {
                    sym::min_align_of_val => align.bytes(),
                    sym::size_of_val => size.bytes(),
                    _ => bug!(),
                };

                self.write_scalar(Scalar::from_target_usize(result, self), dest)?;
            }

            sym::pref_align_of
            | sym::needs_drop
            | sym::type_id
            | sym::type_name
            | sym::variant_count => {
                let gid = GlobalId { instance, promoted: None };
                let ty = match intrinsic_name {
                    sym::pref_align_of | sym::variant_count => self.tcx.types.usize,
                    sym::needs_drop => self.tcx.types.bool,
                    sym::type_id => self.tcx.types.u128,
                    sym::type_name => Ty::new_static_str(self.tcx.tcx),
                    _ => bug!(),
                };
                let val = self
                    .ctfe_query(|tcx| tcx.const_eval_global_id(self.typing_env, gid, tcx.span))?;
                let val = self.const_val_to_op(val, ty, Some(dest.layout))?;
                self.copy_op(&val, dest)?;
            }

            sym::fadd_algebraic
            | sym::fsub_algebraic
            | sym::fmul_algebraic
            | sym::fdiv_algebraic
            | sym::frem_algebraic => {
                let a = self.read_immediate(&args[0])?;
                let b = self.read_immediate(&args[1])?;

                let op = match intrinsic_name {
                    sym::fadd_algebraic => BinOp::Add,
                    sym::fsub_algebraic => BinOp::Sub,
                    sym::fmul_algebraic => BinOp::Mul,
                    sym::fdiv_algebraic => BinOp::Div,
                    sym::frem_algebraic => BinOp::Rem,

                    _ => bug!(),
                };

                let res = self.binary_op(op, &a, &b)?;
                // `binary_op` already called `generate_nan` if needed.
                let res = M::apply_float_nondet(self, res)?;
                self.write_immediate(*res, dest)?;
            }

            sym::ctpop
            | sym::cttz
            | sym::cttz_nonzero
            | sym::ctlz
            | sym::ctlz_nonzero
            | sym::bswap
            | sym::bitreverse => {
                let ty = instance_args.type_at(0);
                let layout = self.layout_of(ty)?;
                let val = self.read_scalar(&args[0])?;

                let out_val = self.numeric_intrinsic(intrinsic_name, val, layout, dest.layout)?;
                self.write_scalar(out_val, dest)?;
            }
            sym::saturating_add | sym::saturating_sub => {
                let l = self.read_immediate(&args[0])?;
                let r = self.read_immediate(&args[1])?;
                let val = self.saturating_arith(
                    if intrinsic_name == sym::saturating_add { BinOp::Add } else { BinOp::Sub },
                    &l,
                    &r,
                )?;
                self.write_scalar(val, dest)?;
            }
            sym::discriminant_value => {
                let place = self.deref_pointer(&args[0])?;
                let variant = self.read_discriminant(&place)?;
                let discr = self.discriminant_for_variant(place.layout.ty, variant)?;
                self.write_immediate(*discr, dest)?;
            }
            sym::exact_div => {
                let l = self.read_immediate(&args[0])?;
                let r = self.read_immediate(&args[1])?;
                self.exact_div(&l, &r, dest)?;
            }
            sym::rotate_left | sym::rotate_right => {
                // rotate_left: (X << (S % BW)) | (X >> ((BW - S) % BW))
                // rotate_right: (X << ((BW - S) % BW)) | (X >> (S % BW))
                let layout_val = self.layout_of(instance_args.type_at(0))?;
                let val = self.read_scalar(&args[0])?;
                let val_bits = val.to_bits(layout_val.size)?; // sign is ignored here

                let layout_raw_shift = self.layout_of(self.tcx.types.u32)?;
                let raw_shift = self.read_scalar(&args[1])?;
                let raw_shift_bits = raw_shift.to_bits(layout_raw_shift.size)?;

                let width_bits = u128::from(layout_val.size.bits());
                let shift_bits = raw_shift_bits % width_bits;
                let inv_shift_bits = (width_bits - shift_bits) % width_bits;
                let result_bits = if intrinsic_name == sym::rotate_left {
                    (val_bits << shift_bits) | (val_bits >> inv_shift_bits)
                } else {
                    (val_bits >> shift_bits) | (val_bits << inv_shift_bits)
                };
                let truncated_bits = layout_val.size.truncate(result_bits);
                let result = Scalar::from_uint(truncated_bits, layout_val.size);
                self.write_scalar(result, dest)?;
            }
            sym::copy => {
                self.copy_intrinsic(&args[0], &args[1], &args[2], /*nonoverlapping*/ false)?;
            }
            sym::write_bytes => {
                self.write_bytes_intrinsic(&args[0], &args[1], &args[2], "write_bytes")?;
            }
            sym::compare_bytes => {
                let result = self.compare_bytes_intrinsic(&args[0], &args[1], &args[2])?;
                self.write_scalar(result, dest)?;
            }
            sym::arith_offset => {
                let ptr = self.read_pointer(&args[0])?;
                let offset_count = self.read_target_isize(&args[1])?;
                let pointee_ty = instance_args.type_at(0);

                let pointee_size = i64::try_from(self.layout_of(pointee_ty)?.size.bytes()).unwrap();
                let offset_bytes = offset_count.wrapping_mul(pointee_size);
                let offset_ptr = ptr.wrapping_signed_offset(offset_bytes, self);
                self.write_pointer(offset_ptr, dest)?;
            }
            sym::ptr_offset_from | sym::ptr_offset_from_unsigned => {
                let a = self.read_pointer(&args[0])?;
                let b = self.read_pointer(&args[1])?;

                let usize_layout = self.layout_of(self.tcx.types.usize)?;
                let isize_layout = self.layout_of(self.tcx.types.isize)?;

                // Get offsets for both that are at least relative to the same base.
                // With `OFFSET_IS_ADDR` this is trivial; without it we need either
                // two integers or two pointers into the same allocation.
                let (a_offset, b_offset, is_addr) = if M::Provenance::OFFSET_IS_ADDR {
                    (a.addr().bytes(), b.addr().bytes(), /*is_addr*/ true)
                } else {
                    match (self.ptr_try_get_alloc_id(a, 0), self.ptr_try_get_alloc_id(b, 0)) {
                        (Err(a), Err(b)) => {
                            // Neither pointer points to an allocation, so they are both absolute.
                            (a, b, /*is_addr*/ true)
                        }
                        (Ok((a_alloc_id, a_offset, _)), Ok((b_alloc_id, b_offset, _)))
                            if a_alloc_id == b_alloc_id =>
                        {
                            // Found allocation for both, and it's the same.
                            // Use these offsets for distance calculation.
                            (a_offset.bytes(), b_offset.bytes(), /*is_addr*/ false)
                        }
                        _ => {
                            // Not into the same allocation -- this is UB.
                            throw_ub_custom!(
                                fluent::const_eval_offset_from_different_allocations,
                                name = intrinsic_name,
                            );
                        }
                    }
                };

                // Compute distance: a - b.
                let dist = {
                    // Addresses are unsigned, so this is a `usize` computation. We have to do the
                    // overflow check separately anyway.
                    let (val, overflowed) = {
                        let a_offset = ImmTy::from_uint(a_offset, usize_layout);
                        let b_offset = ImmTy::from_uint(b_offset, usize_layout);
                        self.binary_op(BinOp::SubWithOverflow, &a_offset, &b_offset)?
                            .to_scalar_pair()
                    };
                    if overflowed.to_bool()? {
                        // a < b
                        if intrinsic_name == sym::ptr_offset_from_unsigned {
                            throw_ub_custom!(
                                fluent::const_eval_offset_from_unsigned_overflow,
                                a_offset = a_offset,
                                b_offset = b_offset,
                                is_addr = is_addr,
                            );
                        }
                        // The signed form of the intrinsic allows this. If we interpret the
                        // difference as isize, we'll get the proper signed difference. If that
                        // seems *positive* or equal to isize::MIN, they were more than isize::MAX apart.
                        let dist = val.to_target_isize(self)?;
                        if dist >= 0 || i128::from(dist) == self.pointer_size().signed_int_min() {
                            throw_ub_custom!(
                                fluent::const_eval_offset_from_underflow,
                                name = intrinsic_name,
                            );
                        }
                        dist
                    } else {
                        // b >= a
                        let dist = val.to_target_isize(self)?;
                        // If converting to isize produced a *negative* result, we had an overflow
                        // because they were more than isize::MAX apart.
                        if dist < 0 {
                            throw_ub_custom!(
                                fluent::const_eval_offset_from_overflow,
                                name = intrinsic_name,
                            );
                        }
                        dist
                    }
                };

                // Check that the memory between them is dereferenceable at all, starting from the
                // origin pointer: `dist` is `a - b`, so it is based on `b`.
                self.check_ptr_access_signed(b, dist, CheckInAllocMsg::Dereferenceable)
                    .map_err_kind(|_| {
                        // This could mean they point to different allocations, or they point to the same allocation
                        // but not the entire range between the pointers is in-bounds.
                        if let Ok((a_alloc_id, ..)) = self.ptr_try_get_alloc_id(a, 0)
                            && let Ok((b_alloc_id, ..)) = self.ptr_try_get_alloc_id(b, 0)
                            && a_alloc_id == b_alloc_id
                        {
                            err_ub_custom!(
                                fluent::const_eval_offset_from_out_of_bounds,
                                name = intrinsic_name,
                            )
                        } else {
                            err_ub_custom!(
                                fluent::const_eval_offset_from_different_allocations,
                                name = intrinsic_name,
                            )
                        }
                    })?;
                // Then check that this is also dereferenceable from `a`. This ensures that they are
                // derived from the same allocation.
                self.check_ptr_access_signed(
                    a,
                    dist.checked_neg().unwrap(), // i64::MIN is impossible as no allocation can be that large
                    CheckInAllocMsg::Dereferenceable,
                )
                .map_err_kind(|_| {
                    // Make the error more specific.
                    err_ub_custom!(
                        fluent::const_eval_offset_from_different_allocations,
                        name = intrinsic_name,
                    )
                })?;

                // Perform division by size to compute return value.
                let ret_layout = if intrinsic_name == sym::ptr_offset_from_unsigned {
                    assert!(0 <= dist && dist <= self.target_isize_max());
                    usize_layout
                } else {
                    assert!(self.target_isize_min() <= dist && dist <= self.target_isize_max());
                    isize_layout
                };
                let pointee_layout = self.layout_of(instance_args.type_at(0))?;
                // If ret_layout is unsigned, we checked that so is the distance, so we are good.
                let val = ImmTy::from_int(dist, ret_layout);
                let size = ImmTy::from_int(pointee_layout.size.bytes(), ret_layout);
                self.exact_div(&val, &size, dest)?;
            }

            sym::assert_inhabited
            | sym::assert_zero_valid
            | sym::assert_mem_uninitialized_valid => {
                let ty = instance.args.type_at(0);
                let requirement = ValidityRequirement::from_intrinsic(intrinsic_name).unwrap();

                let should_panic = !self
                    .tcx
                    .check_validity_requirement((requirement, self.typing_env.as_query_input(ty)))
                    .map_err(|_| err_inval!(TooGeneric))?;

                if should_panic {
                    let layout = self.layout_of(ty)?;

                    let msg = match requirement {
                        // For *all* intrinsics we first check `is_uninhabited` to give a more specific
                        // error message.
                        _ if layout.is_uninhabited() => format!(
                            "aborted execution: attempted to instantiate uninhabited type `{ty}`"
                        ),
                        ValidityRequirement::Inhabited => bug!("handled earlier"),
                        ValidityRequirement::Zero => format!(
                            "aborted execution: attempted to zero-initialize type `{ty}`, which is invalid"
                        ),
                        ValidityRequirement::UninitMitigated0x01Fill => format!(
                            "aborted execution: attempted to leave type `{ty}` uninitialized, which is invalid"
                        ),
                        ValidityRequirement::Uninit => bug!("assert_uninit_valid doesn't exist"),
                    };

                    M::panic_nounwind(self, &msg)?;
                    // Skip the `return_to_block` at the end (we panicked, we do not return).
                    return interp_ok(true);
                }
            }
            sym::simd_insert => {
                let index = u64::from(self.read_scalar(&args[1])?.to_u32()?);
                let elem = &args[2];
                let (input, input_len) = self.project_to_simd(&args[0])?;
                let (dest, dest_len) = self.project_to_simd(dest)?;
                assert_eq!(input_len, dest_len, "Return vector length must match input length");
                // Bounds are not checked by typeck so we have to do it ourselves.
                if index >= input_len {
                    throw_ub_format!(
                        "`simd_insert` index {index} is out-of-bounds of vector with length {input_len}"
                    );
                }

                for i in 0..dest_len {
                    let place = self.project_index(&dest, i)?;
                    let value =
                        if i == index { elem.clone() } else { self.project_index(&input, i)? };
                    self.copy_op(&value, &place)?;
                }
            }
            sym::simd_extract => {
                let index = u64::from(self.read_scalar(&args[1])?.to_u32()?);
                let (input, input_len) = self.project_to_simd(&args[0])?;
                // Bounds are not checked by typeck so we have to do it ourselves.
                if index >= input_len {
                    throw_ub_format!(
                        "`simd_extract` index {index} is out-of-bounds of vector with length {input_len}"
                    );
                }
                self.copy_op(&self.project_index(&input, index)?, dest)?;
            }
            sym::black_box => {
                // These just return their argument
                self.copy_op(&args[0], dest)?;
            }
            sym::raw_eq => {
                let result = self.raw_eq_intrinsic(&args[0], &args[1])?;
                self.write_scalar(result, dest)?;
            }
            sym::typed_swap_nonoverlapping => {
                self.typed_swap_nonoverlapping_intrinsic(&args[0], &args[1])?;
            }

            sym::vtable_size => {
                let ptr = self.read_pointer(&args[0])?;
                // `None` because we don't know which trait to expect here; any vtable is okay.
                let (size, _align) = self.get_vtable_size_and_align(ptr, None)?;
                self.write_scalar(Scalar::from_target_usize(size.bytes(), self), dest)?;
            }
            sym::vtable_align => {
                let ptr = self.read_pointer(&args[0])?;
                // `None` because we don't know which trait to expect here; any vtable is okay.
                let (_size, align) = self.get_vtable_size_and_align(ptr, None)?;
                self.write_scalar(Scalar::from_target_usize(align.bytes(), self), dest)?;
            }

            sym::minnumf16 => self.float_min_intrinsic::<Half>(args, dest)?,
            sym::minnumf32 => self.float_min_intrinsic::<Single>(args, dest)?,
            sym::minnumf64 => self.float_min_intrinsic::<Double>(args, dest)?,
            sym::minnumf128 => self.float_min_intrinsic::<Quad>(args, dest)?,

            sym::minimumf16 => self.float_minimum_intrinsic::<Half>(args, dest)?,
            sym::minimumf32 => self.float_minimum_intrinsic::<Single>(args, dest)?,
            sym::minimumf64 => self.float_minimum_intrinsic::<Double>(args, dest)?,
            sym::minimumf128 => self.float_minimum_intrinsic::<Quad>(args, dest)?,

            sym::maxnumf16 => self.float_max_intrinsic::<Half>(args, dest)?,
            sym::maxnumf32 => self.float_max_intrinsic::<Single>(args, dest)?,
            sym::maxnumf64 => self.float_max_intrinsic::<Double>(args, dest)?,
            sym::maxnumf128 => self.float_max_intrinsic::<Quad>(args, dest)?,

            sym::maximumf16 => self.float_maximum_intrinsic::<Half>(args, dest)?,
            sym::maximumf32 => self.float_maximum_intrinsic::<Single>(args, dest)?,
            sym::maximumf64 => self.float_maximum_intrinsic::<Double>(args, dest)?,
            sym::maximumf128 => self.float_maximum_intrinsic::<Quad>(args, dest)?,

            sym::copysignf16 => self.float_copysign_intrinsic::<Half>(args, dest)?,
            sym::copysignf32 => self.float_copysign_intrinsic::<Single>(args, dest)?,
            sym::copysignf64 => self.float_copysign_intrinsic::<Double>(args, dest)?,
            sym::copysignf128 => self.float_copysign_intrinsic::<Quad>(args, dest)?,

            sym::fabsf16 => self.float_abs_intrinsic::<Half>(args, dest)?,
            sym::fabsf32 => self.float_abs_intrinsic::<Single>(args, dest)?,
            sym::fabsf64 => self.float_abs_intrinsic::<Double>(args, dest)?,
            sym::fabsf128 => self.float_abs_intrinsic::<Quad>(args, dest)?,

            // Unsupported intrinsic: skip the return_to_block below.
            _ => return interp_ok(false),
        }

        trace!("{:?}", self.dump_place(&dest.clone().into()));
        self.return_to_block(ret)?;
        interp_ok(true)
    }

    pub(super) fn eval_nondiverging_intrinsic(
        &mut self,
        intrinsic: &NonDivergingIntrinsic<'tcx>,
    ) -> InterpResult<'tcx> {
        match intrinsic {
            NonDivergingIntrinsic::Assume(op) => {
                let op = self.eval_operand(op, None)?;
                let cond = self.read_scalar(&op)?.to_bool()?;
                if !cond {
                    throw_ub_custom!(fluent::const_eval_assume_false);
                }
                interp_ok(())
            }
            NonDivergingIntrinsic::CopyNonOverlapping(mir::CopyNonOverlapping {
                count,
                src,
                dst,
            }) => {
                let src = self.eval_operand(src, None)?;
                let dst = self.eval_operand(dst, None)?;
                let count = self.eval_operand(count, None)?;
                self.copy_intrinsic(&src, &dst, &count, /* nonoverlapping */ true)
            }
        }
    }

    pub fn numeric_intrinsic(
        &self,
        name: Symbol,
        val: Scalar<M::Provenance>,
        layout: TyAndLayout<'tcx>,
        ret_layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>> {
        assert!(layout.ty.is_integral(), "invalid type for numeric intrinsic: {}", layout.ty);
        let bits = val.to_bits(layout.size)?; // these operations all ignore the sign
        let extra = 128 - u128::from(layout.size.bits());
        let bits_out = match name {
            sym::ctpop => u128::from(bits.count_ones()),
            sym::ctlz_nonzero | sym::cttz_nonzero if bits == 0 => {
                throw_ub_custom!(fluent::const_eval_call_nonzero_intrinsic, name = name,);
            }
            sym::ctlz | sym::ctlz_nonzero => u128::from(bits.leading_zeros()) - extra,
            sym::cttz | sym::cttz_nonzero => u128::from((bits << extra).trailing_zeros()) - extra,
            sym::bswap => {
                assert_eq!(layout, ret_layout);
                (bits << extra).swap_bytes()
            }
            sym::bitreverse => {
                assert_eq!(layout, ret_layout);
                (bits << extra).reverse_bits()
            }
            _ => bug!("not a numeric intrinsic: {}", name),
        };
        interp_ok(Scalar::from_uint(bits_out, ret_layout.size))
    }

    pub fn exact_div(
        &mut self,
        a: &ImmTy<'tcx, M::Provenance>,
        b: &ImmTy<'tcx, M::Provenance>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        assert_eq!(a.layout.ty, b.layout.ty);
        assert_matches!(a.layout.ty.kind(), ty::Int(..) | ty::Uint(..));

        // Performs an exact division, resulting in undefined behavior where
        // `x % y != 0` or `y == 0` or `x == T::MIN && y == -1`.
        // First, check x % y != 0 (or if that computation overflows).
        let rem = self.binary_op(BinOp::Rem, a, b)?;
        // sign does not matter for 0 test, so `to_bits` is fine
        if rem.to_scalar().to_bits(a.layout.size)? != 0 {
            throw_ub_custom!(
                fluent::const_eval_exact_div_has_remainder,
                a = format!("{a}"),
                b = format!("{b}")
            )
        }
        // `Rem` says this is all right, so we can let `Div` do its job.
        let res = self.binary_op(BinOp::Div, a, b)?;
        self.write_immediate(*res, dest)
    }

    pub fn saturating_arith(
        &self,
        mir_op: BinOp,
        l: &ImmTy<'tcx, M::Provenance>,
        r: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>> {
        assert_eq!(l.layout.ty, r.layout.ty);
        assert_matches!(l.layout.ty.kind(), ty::Int(..) | ty::Uint(..));
        assert_matches!(mir_op, BinOp::Add | BinOp::Sub);

        let (val, overflowed) =
            self.binary_op(mir_op.wrapping_to_overflowing().unwrap(), l, r)?.to_scalar_pair();
        interp_ok(if overflowed.to_bool()? {
            let size = l.layout.size;
            if l.layout.backend_repr.is_signed() {
                // For signed ints the saturated value depends on the sign of the first
                // term since the sign of the second term can be inferred from this and
                // the fact that the operation has overflowed (if either is 0 no
                // overflow can occur)
                let first_term: i128 = l.to_scalar().to_int(l.layout.size)?;
                if first_term >= 0 {
                    // Negative overflow not possible since the positive first term
                    // can only increase an (in range) negative term for addition
                    // or corresponding negated positive term for subtraction.
                    Scalar::from_int(size.signed_int_max(), size)
                } else {
                    // Positive overflow not possible for similar reason.
                    Scalar::from_int(size.signed_int_min(), size)
                }
            } else {
                // unsigned
                if matches!(mir_op, BinOp::Add) {
                    // max unsigned
                    Scalar::from_uint(size.unsigned_int_max(), size)
                } else {
                    // underflow to 0
                    Scalar::from_uint(0u128, size)
                }
            }
        } else {
            val
        })
    }

    /// Offsets a pointer by some multiple of its type, returning an error if the pointer leaves its
    /// allocation.
    pub fn ptr_offset_inbounds(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        offset_bytes: i64,
    ) -> InterpResult<'tcx, Pointer<Option<M::Provenance>>> {
        // The offset must be in bounds starting from `ptr`.
        self.check_ptr_access_signed(
            ptr,
            offset_bytes,
            CheckInAllocMsg::InboundsPointerArithmetic,
        )?;
        // This also implies that there is no overflow, so we are done.
        interp_ok(ptr.wrapping_signed_offset(offset_bytes, self))
    }

    /// Copy `count*size_of::<T>()` many bytes from `*src` to `*dst`.
    pub(crate) fn copy_intrinsic(
        &mut self,
        src: &OpTy<'tcx, <M as Machine<'tcx>>::Provenance>,
        dst: &OpTy<'tcx, <M as Machine<'tcx>>::Provenance>,
        count: &OpTy<'tcx, <M as Machine<'tcx>>::Provenance>,
        nonoverlapping: bool,
    ) -> InterpResult<'tcx> {
        let count = self.read_target_usize(count)?;
        let layout = self.layout_of(src.layout.ty.builtin_deref(true).unwrap())?;
        let (size, align) = (layout.size, layout.align.abi);

        let size = self.compute_size_in_bytes(size, count).ok_or_else(|| {
            err_ub_custom!(
                fluent::const_eval_size_overflow,
                name = if nonoverlapping { "copy_nonoverlapping" } else { "copy" }
            )
        })?;

        let src = self.read_pointer(src)?;
        let dst = self.read_pointer(dst)?;

        self.check_ptr_align(src, align)?;
        self.check_ptr_align(dst, align)?;

        self.mem_copy(src, dst, size, nonoverlapping)
    }

    /// Does a *typed* swap of `*left` and `*right`.
    fn typed_swap_nonoverlapping_intrinsic(
        &mut self,
        left: &OpTy<'tcx, <M as Machine<'tcx>>::Provenance>,
        right: &OpTy<'tcx, <M as Machine<'tcx>>::Provenance>,
    ) -> InterpResult<'tcx> {
        let left = self.deref_pointer(left)?;
        let right = self.deref_pointer(right)?;
        assert_eq!(left.layout, right.layout);
        assert!(left.layout.is_sized());
        let kind = MemoryKind::Stack;
        let temp = self.allocate(left.layout, kind)?;
        self.copy_op(&left, &temp)?; // checks alignment of `left`

        // We want to always enforce non-overlapping, even if this is a scalar type.
        // Therefore we directly use the underlying `mem_copy` here.
        self.mem_copy(right.ptr(), left.ptr(), left.layout.size, /*nonoverlapping*/ true)?;
        // This means we also need to do the validation of the value that used to be in `right`
        // ourselves. This value is now in `left.` The one that started out in `left` already got
        // validated by the copy above.
        if M::enforce_validity(self, left.layout) {
            self.validate_operand(
                &left.clone().into(),
                M::enforce_validity_recursively(self, left.layout),
                /*reset_provenance_and_padding*/ true,
            )?;
        }

        self.copy_op(&temp, &right)?; // checks alignment of `right`

        self.deallocate_ptr(temp.ptr(), None, kind)?;
        interp_ok(())
    }

    pub fn write_bytes_intrinsic(
        &mut self,
        dst: &OpTy<'tcx, <M as Machine<'tcx>>::Provenance>,
        byte: &OpTy<'tcx, <M as Machine<'tcx>>::Provenance>,
        count: &OpTy<'tcx, <M as Machine<'tcx>>::Provenance>,
        name: &'static str,
    ) -> InterpResult<'tcx> {
        let layout = self.layout_of(dst.layout.ty.builtin_deref(true).unwrap())?;

        let dst = self.read_pointer(dst)?;
        let byte = self.read_scalar(byte)?.to_u8()?;
        let count = self.read_target_usize(count)?;

        // `checked_mul` enforces a too small bound (the correct one would probably be target_isize_max),
        // but no actual allocation can be big enough for the difference to be noticeable.
        let len = self
            .compute_size_in_bytes(layout.size, count)
            .ok_or_else(|| err_ub_custom!(fluent::const_eval_size_overflow, name = name))?;

        let bytes = std::iter::repeat(byte).take(len.bytes_usize());
        self.write_bytes_ptr(dst, bytes)
    }

    pub(crate) fn compare_bytes_intrinsic(
        &mut self,
        left: &OpTy<'tcx, <M as Machine<'tcx>>::Provenance>,
        right: &OpTy<'tcx, <M as Machine<'tcx>>::Provenance>,
        byte_count: &OpTy<'tcx, <M as Machine<'tcx>>::Provenance>,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>> {
        let left = self.read_pointer(left)?;
        let right = self.read_pointer(right)?;
        let n = Size::from_bytes(self.read_target_usize(byte_count)?);

        let left_bytes = self.read_bytes_ptr_strip_provenance(left, n)?;
        let right_bytes = self.read_bytes_ptr_strip_provenance(right, n)?;

        // `Ordering`'s discriminants are -1/0/+1, so casting does the right thing.
        let result = Ord::cmp(left_bytes, right_bytes) as i32;
        interp_ok(Scalar::from_i32(result))
    }

    pub(crate) fn raw_eq_intrinsic(
        &mut self,
        lhs: &OpTy<'tcx, <M as Machine<'tcx>>::Provenance>,
        rhs: &OpTy<'tcx, <M as Machine<'tcx>>::Provenance>,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>> {
        let layout = self.layout_of(lhs.layout.ty.builtin_deref(true).unwrap())?;
        assert!(layout.is_sized());

        let get_bytes = |this: &InterpCx<'tcx, M>,
                         op: &OpTy<'tcx, <M as Machine<'tcx>>::Provenance>|
         -> InterpResult<'tcx, &[u8]> {
            let ptr = this.read_pointer(op)?;
            this.check_ptr_align(ptr, layout.align.abi)?;
            let Some(alloc_ref) = self.get_ptr_alloc(ptr, layout.size)? else {
                // zero-sized access
                return interp_ok(&[]);
            };
            alloc_ref.get_bytes_strip_provenance()
        };

        let lhs_bytes = get_bytes(self, lhs)?;
        let rhs_bytes = get_bytes(self, rhs)?;
        interp_ok(Scalar::from_bool(lhs_bytes == rhs_bytes))
    }

    fn float_min_intrinsic<F>(
        &mut self,
        args: &[OpTy<'tcx, M::Provenance>],
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ()>
    where
        F: rustc_apfloat::Float + rustc_apfloat::FloatConvert<F> + Into<Scalar<M::Provenance>>,
    {
        let a: F = self.read_scalar(&args[0])?.to_float()?;
        let b: F = self.read_scalar(&args[1])?.to_float()?;
        let res = if a == b {
            // They are definitely not NaN (those are never equal), but they could be `+0` and `-0`.
            // Let the machine decide which one to return.
            M::equal_float_min_max(self, a, b)
        } else {
            self.adjust_nan(a.min(b), &[a, b])
        };
        self.write_scalar(res, dest)?;
        interp_ok(())
    }

    fn float_max_intrinsic<F>(
        &mut self,
        args: &[OpTy<'tcx, M::Provenance>],
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ()>
    where
        F: rustc_apfloat::Float + rustc_apfloat::FloatConvert<F> + Into<Scalar<M::Provenance>>,
    {
        let a: F = self.read_scalar(&args[0])?.to_float()?;
        let b: F = self.read_scalar(&args[1])?.to_float()?;
        let res = if a == b {
            // They are definitely not NaN (those are never equal), but they could be `+0` and `-0`.
            // Let the machine decide which one to return.
            M::equal_float_min_max(self, a, b)
        } else {
            self.adjust_nan(a.max(b), &[a, b])
        };
        self.write_scalar(res, dest)?;
        interp_ok(())
    }

    fn float_minimum_intrinsic<F>(
        &mut self,
        args: &[OpTy<'tcx, M::Provenance>],
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ()>
    where
        F: rustc_apfloat::Float + rustc_apfloat::FloatConvert<F> + Into<Scalar<M::Provenance>>,
    {
        let a: F = self.read_scalar(&args[0])?.to_float()?;
        let b: F = self.read_scalar(&args[1])?.to_float()?;
        let res = a.minimum(b);
        let res = self.adjust_nan(res, &[a, b]);
        self.write_scalar(res, dest)?;
        interp_ok(())
    }

    fn float_maximum_intrinsic<F>(
        &mut self,
        args: &[OpTy<'tcx, M::Provenance>],
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ()>
    where
        F: rustc_apfloat::Float + rustc_apfloat::FloatConvert<F> + Into<Scalar<M::Provenance>>,
    {
        let a: F = self.read_scalar(&args[0])?.to_float()?;
        let b: F = self.read_scalar(&args[1])?.to_float()?;
        let res = a.maximum(b);
        let res = self.adjust_nan(res, &[a, b]);
        self.write_scalar(res, dest)?;
        interp_ok(())
    }

    fn float_copysign_intrinsic<F>(
        &mut self,
        args: &[OpTy<'tcx, M::Provenance>],
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ()>
    where
        F: rustc_apfloat::Float + rustc_apfloat::FloatConvert<F> + Into<Scalar<M::Provenance>>,
    {
        let a: F = self.read_scalar(&args[0])?.to_float()?;
        let b: F = self.read_scalar(&args[1])?.to_float()?;
        // bitwise, no NaN adjustments
        self.write_scalar(a.copy_sign(b), dest)?;
        interp_ok(())
    }

    fn float_abs_intrinsic<F>(
        &mut self,
        args: &[OpTy<'tcx, M::Provenance>],
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ()>
    where
        F: rustc_apfloat::Float + rustc_apfloat::FloatConvert<F> + Into<Scalar<M::Provenance>>,
    {
        let x: F = self.read_scalar(&args[0])?.to_float()?;
        // bitwise, no NaN adjustments
        self.write_scalar(x.abs(), dest)?;
        interp_ok(())
    }
}
