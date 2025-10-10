//! Intrinsics and other functions that the interpreter executes without
//! looking at their MIR. Intrinsics/functions supported here are shared by CTFE
//! and miri.

mod simd;

use std::assert_matches::assert_matches;

use rustc_abi::{FieldIdx, HasDataLayout, Size};
use rustc_apfloat::ieee::{Double, Half, Quad, Single};
use rustc_middle::mir::interpret::{CTFE_ALLOC_SALT, read_target_uint, write_target_uint};
use rustc_middle::mir::{self, BinOp, ConstValue, NonDivergingIntrinsic};
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{FloatTy, Ty, TyCtxt};
use rustc_middle::{bug, span_bug, ty};
use rustc_span::{Symbol, sym};
use tracing::trace;

use super::memory::MemoryKind;
use super::util::ensure_monomorphic_enough;
use super::{
    AllocId, CheckInAllocMsg, ImmTy, InterpCx, InterpResult, Machine, OpTy, PlaceTy, Pointer,
    PointerArithmetic, Provenance, Scalar, err_ub_custom, err_unsup_format, interp_ok, throw_inval,
    throw_ub_custom, throw_ub_format,
};
use crate::fluent_generated as fluent;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MulAddType {
    /// Used with `fma` and `simd_fma`, always uses fused-multiply-add
    Fused,
    /// Used with `fmuladd` and `simd_relaxed_fma`, nondeterministically determines whether to use
    /// fma or simple multiply-add
    Nondeterministic,
}

#[derive(Copy, Clone)]
pub(crate) enum MinMax {
    /// The IEEE `Minimum` operation - see `f32::minimum` etc
    /// In particular, `-0.0` is considered smaller than `+0.0` and
    /// if either input is NaN, the result is NaN.
    Minimum,
    /// The IEEE `MinNum` operation - see `f32::min` etc
    /// In particular, if the inputs are `-0.0` and `+0.0`, the result is non-deterministic,
    /// and is one argument is NaN, the other one is returned.
    MinNum,
    /// The IEEE `Maximum` operation - see `f32::maximum` etc
    /// In particular, `-0.0` is considered smaller than `+0.0` and
    /// if either input is NaN, the result is NaN.
    Maximum,
    /// The IEEE `MaxNum` operation - see `f32::max` etc
    /// In particular, if the inputs are `-0.0` and `+0.0`, the result is non-deterministic,
    /// and is one argument is NaN, the other one is returned.
    MaxNum,
}

/// Directly returns an `Allocation` containing an absolute path representation of the given type.
pub(crate) fn alloc_type_name<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> (AllocId, u64) {
    let path = crate::util::type_name(tcx, ty);
    let bytes = path.into_bytes();
    let len = bytes.len().try_into().unwrap();
    (tcx.allocate_bytes_dedup(bytes, CTFE_ALLOC_SALT), len)
}
impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    /// Generates a value of `TypeId` for `ty` in-place.
    fn write_type_id(
        &mut self,
        ty: Ty<'tcx>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ()> {
        let tcx = self.tcx;
        let type_id_hash = tcx.type_id_hash(ty).as_u128();
        let op = self.const_val_to_op(
            ConstValue::Scalar(Scalar::from_u128(type_id_hash)),
            tcx.types.u128,
            None,
        )?;
        self.copy_op_allow_transmute(&op, dest)?;

        // Give the each pointer-sized chunk provenance that knows about the type id.
        // Here we rely on `TypeId` being a newtype around an array of pointers, so we
        // first project to its only field and then the array elements.
        let alloc_id = tcx.reserve_and_set_type_id_alloc(ty);
        let arr = self.project_field(dest, FieldIdx::ZERO)?;
        let mut elem_iter = self.project_array_fields(&arr)?;
        while let Some((_, elem)) = elem_iter.next(self)? {
            // Decorate this part of the hash with provenance; leave the integer part unchanged.
            let hash_fragment = self.read_scalar(&elem)?.to_target_usize(&tcx)?;
            let ptr = Pointer::new(alloc_id.into(), Size::from_bytes(hash_fragment));
            let ptr = self.global_root_pointer(ptr)?;
            let val = Scalar::from_pointer(ptr, &tcx);
            self.write_scalar(val, &elem)?;
        }
        interp_ok(())
    }

    /// Read a value of type `TypeId`, returning the type it represents.
    pub(crate) fn read_type_id(
        &self,
        op: &OpTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Ty<'tcx>> {
        // `TypeId` is a newtype around an array of pointers. All pointers must have the same
        // provenance, and that provenance represents the type.
        let ptr_size = self.pointer_size().bytes_usize();
        let arr = self.project_field(op, FieldIdx::ZERO)?;

        let mut ty_and_hash = None;
        let mut elem_iter = self.project_array_fields(&arr)?;
        while let Some((idx, elem)) = elem_iter.next(self)? {
            let elem = self.read_pointer(&elem)?;
            let (elem_ty, elem_hash) = self.get_ptr_type_id(elem)?;
            // If this is the first element, remember the type and its hash.
            // If this is not the first element, ensure it is consistent with the previous ones.
            let full_hash = match ty_and_hash {
                None => {
                    let hash = self.tcx.type_id_hash(elem_ty).as_u128();
                    let mut hash_bytes = [0u8; 16];
                    write_target_uint(self.data_layout().endian, &mut hash_bytes, hash).unwrap();
                    ty_and_hash = Some((elem_ty, hash_bytes));
                    hash_bytes
                }
                Some((ty, hash_bytes)) => {
                    if ty != elem_ty {
                        throw_ub_format!(
                            "invalid `TypeId` value: not all bytes carry the same type id metadata"
                        );
                    }
                    hash_bytes
                }
            };
            // Ensure the elem_hash matches the corresponding part of the full hash.
            let hash_frag = &full_hash[(idx as usize) * ptr_size..][..ptr_size];
            if read_target_uint(self.data_layout().endian, hash_frag).unwrap() != elem_hash.into() {
                throw_ub_format!(
                    "invalid `TypeId` value: the hash does not match the type id metadata"
                );
            }
        }

        interp_ok(ty_and_hash.unwrap().0)
    }

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

        if intrinsic_name.as_str().starts_with("simd_") {
            return self.eval_simd_intrinsic(intrinsic_name, instance_args, args, dest, ret);
        }

        let tcx = self.tcx.tcx;

        match intrinsic_name {
            sym::type_name => {
                let tp_ty = instance.args.type_at(0);
                ensure_monomorphic_enough(tcx, tp_ty)?;
                let (alloc_id, meta) = alloc_type_name(tcx, tp_ty);
                let val = ConstValue::Slice { alloc_id, meta };
                let val = self.const_val_to_op(val, dest.layout.ty, Some(dest.layout))?;
                self.copy_op(&val, dest)?;
            }
            sym::needs_drop => {
                let tp_ty = instance.args.type_at(0);
                ensure_monomorphic_enough(tcx, tp_ty)?;
                let val = ConstValue::from_bool(tp_ty.needs_drop(tcx, self.typing_env));
                let val = self.const_val_to_op(val, tcx.types.bool, Some(dest.layout))?;
                self.copy_op(&val, dest)?;
            }
            sym::type_id => {
                let tp_ty = instance.args.type_at(0);
                ensure_monomorphic_enough(tcx, tp_ty)?;
                self.write_type_id(tp_ty, dest)?;
            }
            sym::type_id_eq => {
                let a_ty = self.read_type_id(&args[0])?;
                let b_ty = self.read_type_id(&args[1])?;
                self.write_scalar(Scalar::from_bool(a_ty == b_ty), dest)?;
            }
            sym::size_of => {
                let tp_ty = instance.args.type_at(0);
                let layout = self.layout_of(tp_ty)?;
                if !layout.is_sized() {
                    span_bug!(self.cur_span(), "unsized type for `size_of`");
                }
                let val = layout.size.bytes();
                self.write_scalar(Scalar::from_target_usize(val, self), dest)?;
            }
            sym::align_of => {
                let tp_ty = instance.args.type_at(0);
                let layout = self.layout_of(tp_ty)?;
                if !layout.is_sized() {
                    span_bug!(self.cur_span(), "unsized type for `align_of`");
                }
                let val = layout.align.bytes();
                self.write_scalar(Scalar::from_target_usize(val, self), dest)?;
            }
            sym::variant_count => {
                let tp_ty = instance.args.type_at(0);
                let ty = match tp_ty.kind() {
                    // Pattern types have the same number of variants as their base type.
                    // Even if we restrict e.g. which variants are valid, the variants are essentially just uninhabited.
                    // And `Result<(), !>` still has two variants according to `variant_count`.
                    ty::Pat(base, _) => *base,
                    _ => tp_ty,
                };
                let val = match ty.kind() {
                    // Correctly handles non-monomorphic calls, so there is no need for ensure_monomorphic_enough.
                    ty::Adt(adt, _) => {
                        ConstValue::from_target_usize(adt.variants().len() as u64, &tcx)
                    }
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
                    | ty::Dynamic(_, _)
                    | ty::Closure(_, _)
                    | ty::CoroutineClosure(_, _)
                    | ty::Coroutine(_, _)
                    | ty::CoroutineWitness(..)
                    | ty::UnsafeBinder(_)
                    | ty::Never
                    | ty::Tuple(_)
                    | ty::Error(_) => ConstValue::from_target_usize(0u64, &tcx),
                };
                let val = self.const_val_to_op(val, dest.layout.ty, Some(dest.layout))?;
                self.copy_op(&val, dest)?;
            }

            sym::caller_location => {
                let span = self.find_closest_untracked_caller_location();
                let val = self.tcx.span_as_caller_location(span);
                let val =
                    self.const_val_to_op(val, self.tcx.caller_location_ty(), Some(dest.layout))?;
                self.copy_op(&val, dest)?;
            }

            sym::align_of_val | sym::size_of_val => {
                // Avoid `deref_pointer` -- this is not a deref, the ptr does not have to be
                // dereferenceable!
                let place = self.ref_to_mplace(&self.read_immediate(&args[0])?)?;
                let (size, align) = self
                    .size_and_align_of_val(&place)?
                    .ok_or_else(|| err_unsup_format!("`extern type` does not have known layout"))?;

                let result = match intrinsic_name {
                    sym::align_of_val => align.bytes(),
                    sym::size_of_val => size.bytes(),
                    _ => bug!(),
                };

                self.write_scalar(Scalar::from_target_usize(result, self), dest)?;
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

            sym::minnumf16 => self.float_minmax_intrinsic::<Half>(args, MinMax::MinNum, dest)?,
            sym::minnumf32 => self.float_minmax_intrinsic::<Single>(args, MinMax::MinNum, dest)?,
            sym::minnumf64 => self.float_minmax_intrinsic::<Double>(args, MinMax::MinNum, dest)?,
            sym::minnumf128 => self.float_minmax_intrinsic::<Quad>(args, MinMax::MinNum, dest)?,

            sym::minimumf16 => self.float_minmax_intrinsic::<Half>(args, MinMax::Minimum, dest)?,
            sym::minimumf32 => {
                self.float_minmax_intrinsic::<Single>(args, MinMax::Minimum, dest)?
            }
            sym::minimumf64 => {
                self.float_minmax_intrinsic::<Double>(args, MinMax::Minimum, dest)?
            }
            sym::minimumf128 => self.float_minmax_intrinsic::<Quad>(args, MinMax::Minimum, dest)?,

            sym::maxnumf16 => self.float_minmax_intrinsic::<Half>(args, MinMax::MaxNum, dest)?,
            sym::maxnumf32 => self.float_minmax_intrinsic::<Single>(args, MinMax::MaxNum, dest)?,
            sym::maxnumf64 => self.float_minmax_intrinsic::<Double>(args, MinMax::MaxNum, dest)?,
            sym::maxnumf128 => self.float_minmax_intrinsic::<Quad>(args, MinMax::MaxNum, dest)?,

            sym::maximumf16 => self.float_minmax_intrinsic::<Half>(args, MinMax::Maximum, dest)?,
            sym::maximumf32 => {
                self.float_minmax_intrinsic::<Single>(args, MinMax::Maximum, dest)?
            }
            sym::maximumf64 => {
                self.float_minmax_intrinsic::<Double>(args, MinMax::Maximum, dest)?
            }
            sym::maximumf128 => self.float_minmax_intrinsic::<Quad>(args, MinMax::Maximum, dest)?,

            sym::copysignf16 => self.float_copysign_intrinsic::<Half>(args, dest)?,
            sym::copysignf32 => self.float_copysign_intrinsic::<Single>(args, dest)?,
            sym::copysignf64 => self.float_copysign_intrinsic::<Double>(args, dest)?,
            sym::copysignf128 => self.float_copysign_intrinsic::<Quad>(args, dest)?,

            sym::fabsf16 => self.float_abs_intrinsic::<Half>(args, dest)?,
            sym::fabsf32 => self.float_abs_intrinsic::<Single>(args, dest)?,
            sym::fabsf64 => self.float_abs_intrinsic::<Double>(args, dest)?,
            sym::fabsf128 => self.float_abs_intrinsic::<Quad>(args, dest)?,

            sym::floorf16 => self.float_round_intrinsic::<Half>(
                args,
                dest,
                rustc_apfloat::Round::TowardNegative,
            )?,
            sym::floorf32 => self.float_round_intrinsic::<Single>(
                args,
                dest,
                rustc_apfloat::Round::TowardNegative,
            )?,
            sym::floorf64 => self.float_round_intrinsic::<Double>(
                args,
                dest,
                rustc_apfloat::Round::TowardNegative,
            )?,
            sym::floorf128 => self.float_round_intrinsic::<Quad>(
                args,
                dest,
                rustc_apfloat::Round::TowardNegative,
            )?,

            sym::ceilf16 => self.float_round_intrinsic::<Half>(
                args,
                dest,
                rustc_apfloat::Round::TowardPositive,
            )?,
            sym::ceilf32 => self.float_round_intrinsic::<Single>(
                args,
                dest,
                rustc_apfloat::Round::TowardPositive,
            )?,
            sym::ceilf64 => self.float_round_intrinsic::<Double>(
                args,
                dest,
                rustc_apfloat::Round::TowardPositive,
            )?,
            sym::ceilf128 => self.float_round_intrinsic::<Quad>(
                args,
                dest,
                rustc_apfloat::Round::TowardPositive,
            )?,

            sym::truncf16 => {
                self.float_round_intrinsic::<Half>(args, dest, rustc_apfloat::Round::TowardZero)?
            }
            sym::truncf32 => {
                self.float_round_intrinsic::<Single>(args, dest, rustc_apfloat::Round::TowardZero)?
            }
            sym::truncf64 => {
                self.float_round_intrinsic::<Double>(args, dest, rustc_apfloat::Round::TowardZero)?
            }
            sym::truncf128 => {
                self.float_round_intrinsic::<Quad>(args, dest, rustc_apfloat::Round::TowardZero)?
            }

            sym::roundf16 => self.float_round_intrinsic::<Half>(
                args,
                dest,
                rustc_apfloat::Round::NearestTiesToAway,
            )?,
            sym::roundf32 => self.float_round_intrinsic::<Single>(
                args,
                dest,
                rustc_apfloat::Round::NearestTiesToAway,
            )?,
            sym::roundf64 => self.float_round_intrinsic::<Double>(
                args,
                dest,
                rustc_apfloat::Round::NearestTiesToAway,
            )?,
            sym::roundf128 => self.float_round_intrinsic::<Quad>(
                args,
                dest,
                rustc_apfloat::Round::NearestTiesToAway,
            )?,

            sym::round_ties_even_f16 => self.float_round_intrinsic::<Half>(
                args,
                dest,
                rustc_apfloat::Round::NearestTiesToEven,
            )?,
            sym::round_ties_even_f32 => self.float_round_intrinsic::<Single>(
                args,
                dest,
                rustc_apfloat::Round::NearestTiesToEven,
            )?,
            sym::round_ties_even_f64 => self.float_round_intrinsic::<Double>(
                args,
                dest,
                rustc_apfloat::Round::NearestTiesToEven,
            )?,
            sym::round_ties_even_f128 => self.float_round_intrinsic::<Quad>(
                args,
                dest,
                rustc_apfloat::Round::NearestTiesToEven,
            )?,
            sym::fmaf16 => self.float_muladd_intrinsic::<Half>(args, dest, MulAddType::Fused)?,
            sym::fmaf32 => self.float_muladd_intrinsic::<Single>(args, dest, MulAddType::Fused)?,
            sym::fmaf64 => self.float_muladd_intrinsic::<Double>(args, dest, MulAddType::Fused)?,
            sym::fmaf128 => self.float_muladd_intrinsic::<Quad>(args, dest, MulAddType::Fused)?,
            sym::fmuladdf16 => {
                self.float_muladd_intrinsic::<Half>(args, dest, MulAddType::Nondeterministic)?
            }
            sym::fmuladdf32 => {
                self.float_muladd_intrinsic::<Single>(args, dest, MulAddType::Nondeterministic)?
            }
            sym::fmuladdf64 => {
                self.float_muladd_intrinsic::<Double>(args, dest, MulAddType::Nondeterministic)?
            }
            sym::fmuladdf128 => {
                self.float_muladd_intrinsic::<Quad>(args, dest, MulAddType::Nondeterministic)?
            }

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

        let bytes = std::iter::repeat_n(byte, len.bytes_usize());
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

    fn float_minmax<F>(
        &self,
        a: Scalar<M::Provenance>,
        b: Scalar<M::Provenance>,
        op: MinMax,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>>
    where
        F: rustc_apfloat::Float + rustc_apfloat::FloatConvert<F> + Into<Scalar<M::Provenance>>,
    {
        let a: F = a.to_float()?;
        let b: F = b.to_float()?;
        let res = if matches!(op, MinMax::MinNum | MinMax::MaxNum) && a == b {
            // They are definitely not NaN (those are never equal), but they could be `+0` and `-0`.
            // Let the machine decide which one to return.
            M::equal_float_min_max(self, a, b)
        } else {
            let result = match op {
                MinMax::Minimum => a.minimum(b),
                MinMax::MinNum => a.min(b),
                MinMax::Maximum => a.maximum(b),
                MinMax::MaxNum => a.max(b),
            };
            self.adjust_nan(result, &[a, b])
        };

        interp_ok(res.into())
    }

    fn float_minmax_intrinsic<F>(
        &mut self,
        args: &[OpTy<'tcx, M::Provenance>],
        op: MinMax,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ()>
    where
        F: rustc_apfloat::Float + rustc_apfloat::FloatConvert<F> + Into<Scalar<M::Provenance>>,
    {
        let res =
            self.float_minmax::<F>(self.read_scalar(&args[0])?, self.read_scalar(&args[1])?, op)?;
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

    fn float_round<F>(
        &mut self,
        x: Scalar<M::Provenance>,
        mode: rustc_apfloat::Round,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>>
    where
        F: rustc_apfloat::Float + rustc_apfloat::FloatConvert<F> + Into<Scalar<M::Provenance>>,
    {
        let x: F = x.to_float()?;
        let res = x.round_to_integral(mode).value;
        let res = self.adjust_nan(res, &[x]);
        interp_ok(res.into())
    }

    fn float_round_intrinsic<F>(
        &mut self,
        args: &[OpTy<'tcx, M::Provenance>],
        dest: &PlaceTy<'tcx, M::Provenance>,
        mode: rustc_apfloat::Round,
    ) -> InterpResult<'tcx, ()>
    where
        F: rustc_apfloat::Float + rustc_apfloat::FloatConvert<F> + Into<Scalar<M::Provenance>>,
    {
        let res = self.float_round::<F>(self.read_scalar(&args[0])?, mode)?;
        self.write_scalar(res, dest)?;
        interp_ok(())
    }

    fn float_muladd<F>(
        &self,
        a: Scalar<M::Provenance>,
        b: Scalar<M::Provenance>,
        c: Scalar<M::Provenance>,
        typ: MulAddType,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>>
    where
        F: rustc_apfloat::Float + rustc_apfloat::FloatConvert<F> + Into<Scalar<M::Provenance>>,
    {
        let a: F = a.to_float()?;
        let b: F = b.to_float()?;
        let c: F = c.to_float()?;

        let fuse = typ == MulAddType::Fused || M::float_fuse_mul_add(self);

        let res = if fuse { a.mul_add(b, c).value } else { ((a * b).value + c).value };
        let res = self.adjust_nan(res, &[a, b, c]);
        interp_ok(res.into())
    }

    fn float_muladd_intrinsic<F>(
        &mut self,
        args: &[OpTy<'tcx, M::Provenance>],
        dest: &PlaceTy<'tcx, M::Provenance>,
        typ: MulAddType,
    ) -> InterpResult<'tcx, ()>
    where
        F: rustc_apfloat::Float + rustc_apfloat::FloatConvert<F> + Into<Scalar<M::Provenance>>,
    {
        let a = self.read_scalar(&args[0])?;
        let b = self.read_scalar(&args[1])?;
        let c = self.read_scalar(&args[2])?;

        let res = self.float_muladd::<F>(a, b, c, typ)?;
        self.write_scalar(res, dest)?;
        interp_ok(())
    }

    /// Converts `src` from floating point to integer type `dest_ty`
    /// after rounding with mode `round`.
    /// Returns `None` if `f` is NaN or out of range.
    pub fn float_to_int_checked(
        &self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_to: TyAndLayout<'tcx>,
        round: rustc_apfloat::Round,
    ) -> InterpResult<'tcx, Option<ImmTy<'tcx, M::Provenance>>> {
        fn float_to_int_inner<'tcx, F: rustc_apfloat::Float, M: Machine<'tcx>>(
            ecx: &InterpCx<'tcx, M>,
            src: F,
            cast_to: TyAndLayout<'tcx>,
            round: rustc_apfloat::Round,
        ) -> (Scalar<M::Provenance>, rustc_apfloat::Status) {
            let int_size = cast_to.layout.size;
            match cast_to.ty.kind() {
                // Unsigned
                ty::Uint(_) => {
                    let res = src.to_u128_r(int_size.bits_usize(), round, &mut false);
                    (Scalar::from_uint(res.value, int_size), res.status)
                }
                // Signed
                ty::Int(_) => {
                    let res = src.to_i128_r(int_size.bits_usize(), round, &mut false);
                    (Scalar::from_int(res.value, int_size), res.status)
                }
                // Nothing else
                _ => span_bug!(
                    ecx.cur_span(),
                    "attempted float-to-int conversion with non-int output type {}",
                    cast_to.ty,
                ),
            }
        }

        let ty::Float(fty) = src.layout.ty.kind() else {
            bug!("float_to_int_checked: non-float input type {}", src.layout.ty)
        };

        let (val, status) = match fty {
            FloatTy::F16 => float_to_int_inner(self, src.to_scalar().to_f16()?, cast_to, round),
            FloatTy::F32 => float_to_int_inner(self, src.to_scalar().to_f32()?, cast_to, round),
            FloatTy::F64 => float_to_int_inner(self, src.to_scalar().to_f64()?, cast_to, round),
            FloatTy::F128 => float_to_int_inner(self, src.to_scalar().to_f128()?, cast_to, round),
        };

        if status.intersects(
            rustc_apfloat::Status::INVALID_OP
                | rustc_apfloat::Status::OVERFLOW
                | rustc_apfloat::Status::UNDERFLOW,
        ) {
            // Floating point value is NaN (flagged with INVALID_OP) or outside the range
            // of values of the integer type (flagged with OVERFLOW or UNDERFLOW).
            interp_ok(None)
        } else {
            // Floating point value can be represented by the integer type after rounding.
            // The INEXACT flag is ignored on purpose to allow rounding.
            interp_ok(Some(ImmTy::from_scalar(val, cast_to)))
        }
    }
}
