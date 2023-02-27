//! Intrinsics and other functions that the miri engine executes without
//! looking at their MIR. Intrinsics/functions supported here are shared by CTFE
//! and miri.

use rustc_hir::def_id::DefId;
use rustc_middle::mir::{
    self,
    interpret::{
        Allocation, ConstAllocation, ConstValue, GlobalId, InterpResult, PointerArithmetic, Scalar,
    },
    BinOp, NonDivergingIntrinsic,
};
use rustc_middle::ty;
use rustc_middle::ty::layout::{InitKind, LayoutOf as _};
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_span::symbol::{sym, Symbol};
use rustc_target::abi::{Abi, Align, Primitive, Size};

use super::{
    util::ensure_monomorphic_enough, CheckInAllocMsg, ImmTy, InterpCx, Machine, OpTy, PlaceTy,
    Pointer,
};

mod caller_location;

fn numeric_intrinsic<Prov>(name: Symbol, bits: u128, kind: Primitive) -> Scalar<Prov> {
    let size = match kind {
        Primitive::Int(integer, _) => integer.size(),
        _ => bug!("invalid `{}` argument: {:?}", name, bits),
    };
    let extra = 128 - u128::from(size.bits());
    let bits_out = match name {
        sym::ctpop => u128::from(bits.count_ones()),
        sym::ctlz => u128::from(bits.leading_zeros()) - extra,
        sym::cttz => u128::from((bits << extra).trailing_zeros()) - extra,
        sym::bswap => (bits << extra).swap_bytes(),
        sym::bitreverse => (bits << extra).reverse_bits(),
        _ => bug!("not a numeric intrinsic: {}", name),
    };
    Scalar::from_uint(bits_out, size)
}

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
    param_env: ty::ParamEnv<'tcx>,
    def_id: DefId,
    substs: SubstsRef<'tcx>,
) -> InterpResult<'tcx, ConstValue<'tcx>> {
    let tp_ty = substs.type_at(0);
    let name = tcx.item_name(def_id);
    Ok(match name {
        sym::type_name => {
            ensure_monomorphic_enough(tcx, tp_ty)?;
            let alloc = alloc_type_name(tcx, tp_ty);
            ConstValue::Slice { data: alloc, start: 0, end: alloc.inner().len() }
        }
        sym::needs_drop => {
            ensure_monomorphic_enough(tcx, tp_ty)?;
            ConstValue::from_bool(tp_ty.needs_drop(tcx, param_env))
        }
        sym::pref_align_of => {
            // Correctly handles non-monomorphic calls, so there is no need for ensure_monomorphic_enough.
            let layout = tcx.layout_of(param_env.and(tp_ty)).map_err(|e| err_inval!(Layout(e)))?;
            ConstValue::from_target_usize(layout.align.pref.bytes(), &tcx)
        }
        sym::type_id => {
            ensure_monomorphic_enough(tcx, tp_ty)?;
            ConstValue::from_u64(tcx.type_id_hash(tp_ty))
        }
        sym::variant_count => match tp_ty.kind() {
            // Correctly handles non-monomorphic calls, so there is no need for ensure_monomorphic_enough.
            ty::Adt(adt, _) => ConstValue::from_target_usize(adt.variants().len() as u64, &tcx),
            ty::Alias(..) | ty::Param(_) | ty::Placeholder(_) | ty::Infer(_) => {
                throw_inval!(TooGeneric)
            }
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
            | ty::RawPtr(_)
            | ty::Ref(_, _, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(_)
            | ty::Dynamic(_, _, _)
            | ty::Closure(_, _)
            | ty::Generator(_, _, _)
            | ty::GeneratorWitness(_)
            | ty::GeneratorWitnessMIR(_, _)
            | ty::Never
            | ty::Tuple(_)
            | ty::Error(_) => ConstValue::from_target_usize(0u64, &tcx),
        },
        other => bug!("`{}` is not a zero arg intrinsic", other),
    })
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Returns `true` if emulation happened.
    /// Here we implement the intrinsics that are common to all Miri instances; individual machines can add their own
    /// intrinsic handling.
    pub fn emulate_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, M::Provenance>],
        dest: &PlaceTy<'tcx, M::Provenance>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, bool> {
        let substs = instance.substs;
        let intrinsic_name = self.tcx.item_name(instance.def_id());

        // First handle intrinsics without return place.
        let ret = match ret {
            None => match intrinsic_name {
                sym::transmute => throw_ub_format!("transmuting to uninhabited type"),
                sym::abort => M::abort(self, "the program aborted execution".to_owned())?,
                // Unsupported diverging intrinsic.
                _ => return Ok(false),
            },
            Some(p) => p,
        };

        match intrinsic_name {
            sym::caller_location => {
                let span = self.find_closest_untracked_caller_location();
                let location = self.alloc_caller_location_for_span(span);
                self.write_immediate(location.to_ref(self), dest)?;
            }

            sym::min_align_of_val | sym::size_of_val => {
                // Avoid `deref_operand` -- this is not a deref, the ptr does not have to be
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
                    sym::type_id => self.tcx.types.u64,
                    sym::type_name => self.tcx.mk_static_str(),
                    _ => bug!(),
                };
                let val = self.ctfe_query(None, |tcx| {
                    tcx.const_eval_global_id(self.param_env, gid, Some(tcx.span))
                })?;
                let val = self.const_val_to_op(val, ty, Some(dest.layout))?;
                self.copy_op(&val, dest, /*allow_transmute*/ false)?;
            }

            sym::ctpop
            | sym::cttz
            | sym::cttz_nonzero
            | sym::ctlz
            | sym::ctlz_nonzero
            | sym::bswap
            | sym::bitreverse => {
                let ty = substs.type_at(0);
                let layout_of = self.layout_of(ty)?;
                let val = self.read_scalar(&args[0])?;
                let bits = val.to_bits(layout_of.size)?;
                let kind = match layout_of.abi {
                    Abi::Scalar(scalar) => scalar.primitive(),
                    _ => span_bug!(
                        self.cur_span(),
                        "{} called on invalid type {:?}",
                        intrinsic_name,
                        ty
                    ),
                };
                let (nonzero, intrinsic_name) = match intrinsic_name {
                    sym::cttz_nonzero => (true, sym::cttz),
                    sym::ctlz_nonzero => (true, sym::ctlz),
                    other => (false, other),
                };
                if nonzero && bits == 0 {
                    throw_ub_format!("`{}_nonzero` called on 0", intrinsic_name);
                }
                let out_val = numeric_intrinsic(intrinsic_name, bits, kind);
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
                let place = self.deref_operand(&args[0])?;
                let discr_val = self.read_discriminant(&place.into())?.0;
                self.write_scalar(discr_val, dest)?;
            }
            sym::exact_div => {
                let l = self.read_immediate(&args[0])?;
                let r = self.read_immediate(&args[1])?;
                self.exact_div(&l, &r, dest)?;
            }
            sym::unchecked_shl
            | sym::unchecked_shr
            | sym::unchecked_add
            | sym::unchecked_sub
            | sym::unchecked_mul
            | sym::unchecked_div
            | sym::unchecked_rem => {
                let l = self.read_immediate(&args[0])?;
                let r = self.read_immediate(&args[1])?;
                let bin_op = match intrinsic_name {
                    sym::unchecked_shl => BinOp::Shl,
                    sym::unchecked_shr => BinOp::Shr,
                    sym::unchecked_add => BinOp::Add,
                    sym::unchecked_sub => BinOp::Sub,
                    sym::unchecked_mul => BinOp::Mul,
                    sym::unchecked_div => BinOp::Div,
                    sym::unchecked_rem => BinOp::Rem,
                    _ => bug!(),
                };
                let (val, overflowed, _ty) = self.overflowing_binary_op(bin_op, &l, &r)?;
                if overflowed {
                    let layout = self.layout_of(substs.type_at(0))?;
                    let r_val = r.to_scalar().to_bits(layout.size)?;
                    if let sym::unchecked_shl | sym::unchecked_shr = intrinsic_name {
                        throw_ub_format!("overflowing shift by {} in `{}`", r_val, intrinsic_name);
                    } else {
                        throw_ub_format!("overflow executing `{}`", intrinsic_name);
                    }
                }
                self.write_scalar(val, dest)?;
            }
            sym::rotate_left | sym::rotate_right => {
                // rotate_left: (X << (S % BW)) | (X >> ((BW - S) % BW))
                // rotate_right: (X << ((BW - S) % BW)) | (X >> (S % BW))
                let layout = self.layout_of(substs.type_at(0))?;
                let val = self.read_scalar(&args[0])?;
                let val_bits = val.to_bits(layout.size)?;
                let raw_shift = self.read_scalar(&args[1])?;
                let raw_shift_bits = raw_shift.to_bits(layout.size)?;
                let width_bits = u128::from(layout.size.bits());
                let shift_bits = raw_shift_bits % width_bits;
                let inv_shift_bits = (width_bits - shift_bits) % width_bits;
                let result_bits = if intrinsic_name == sym::rotate_left {
                    (val_bits << shift_bits) | (val_bits >> inv_shift_bits)
                } else {
                    (val_bits >> shift_bits) | (val_bits << inv_shift_bits)
                };
                let truncated_bits = self.truncate(result_bits, layout);
                let result = Scalar::from_uint(truncated_bits, layout.size);
                self.write_scalar(result, dest)?;
            }
            sym::copy => {
                self.copy_intrinsic(&args[0], &args[1], &args[2], /*nonoverlapping*/ false)?;
            }
            sym::write_bytes => {
                self.write_bytes_intrinsic(&args[0], &args[1], &args[2])?;
            }
            sym::offset => {
                let ptr = self.read_pointer(&args[0])?;
                let offset_count = self.read_target_isize(&args[1])?;
                let pointee_ty = substs.type_at(0);

                let offset_ptr = self.ptr_offset_inbounds(ptr, pointee_ty, offset_count)?;
                self.write_pointer(offset_ptr, dest)?;
            }
            sym::arith_offset => {
                let ptr = self.read_pointer(&args[0])?;
                let offset_count = self.read_target_isize(&args[1])?;
                let pointee_ty = substs.type_at(0);

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
                let (a_offset, b_offset) =
                    match (self.ptr_try_get_alloc_id(a), self.ptr_try_get_alloc_id(b)) {
                        (Err(a), Err(b)) => {
                            // Neither pointer points to an allocation.
                            // If these are inequal or null, this *will* fail the deref check below.
                            (a, b)
                        }
                        (Err(_), _) | (_, Err(_)) => {
                            // We managed to find a valid allocation for one pointer, but not the other.
                            // That means they are definitely not pointing to the same allocation.
                            throw_ub_format!(
                                "`{}` called on pointers into different allocations",
                                intrinsic_name
                            );
                        }
                        (Ok((a_alloc_id, a_offset, _)), Ok((b_alloc_id, b_offset, _))) => {
                            // Found allocation for both. They must be into the same allocation.
                            if a_alloc_id != b_alloc_id {
                                throw_ub_format!(
                                    "`{}` called on pointers into different allocations",
                                    intrinsic_name
                                );
                            }
                            // Use these offsets for distance calculation.
                            (a_offset.bytes(), b_offset.bytes())
                        }
                    };

                // Compute distance.
                let dist = {
                    // Addresses are unsigned, so this is a `usize` computation. We have to do the
                    // overflow check separately anyway.
                    let (val, overflowed, _ty) = {
                        let a_offset = ImmTy::from_uint(a_offset, usize_layout);
                        let b_offset = ImmTy::from_uint(b_offset, usize_layout);
                        self.overflowing_binary_op(BinOp::Sub, &a_offset, &b_offset)?
                    };
                    if overflowed {
                        // a < b
                        if intrinsic_name == sym::ptr_offset_from_unsigned {
                            throw_ub_format!(
                                "`{}` called when first pointer has smaller offset than second: {} < {}",
                                intrinsic_name,
                                a_offset,
                                b_offset,
                            );
                        }
                        // The signed form of the intrinsic allows this. If we interpret the
                        // difference as isize, we'll get the proper signed difference. If that
                        // seems *positive*, they were more than isize::MAX apart.
                        let dist = val.to_target_isize(self)?;
                        if dist >= 0 {
                            throw_ub_format!(
                                "`{}` called when first pointer is too far before second",
                                intrinsic_name
                            );
                        }
                        dist
                    } else {
                        // b >= a
                        let dist = val.to_target_isize(self)?;
                        // If converting to isize produced a *negative* result, we had an overflow
                        // because they were more than isize::MAX apart.
                        if dist < 0 {
                            throw_ub_format!(
                                "`{}` called when first pointer is too far ahead of second",
                                intrinsic_name
                            );
                        }
                        dist
                    }
                };

                // Check that the range between them is dereferenceable ("in-bounds or one past the
                // end of the same allocation"). This is like the check in ptr_offset_inbounds.
                let min_ptr = if dist >= 0 { b } else { a };
                self.check_ptr_access_align(
                    min_ptr,
                    Size::from_bytes(dist.unsigned_abs()),
                    Align::ONE,
                    CheckInAllocMsg::OffsetFromTest,
                )?;

                // Perform division by size to compute return value.
                let ret_layout = if intrinsic_name == sym::ptr_offset_from_unsigned {
                    assert!(0 <= dist && dist <= self.target_isize_max());
                    usize_layout
                } else {
                    assert!(self.target_isize_min() <= dist && dist <= self.target_isize_max());
                    isize_layout
                };
                let pointee_layout = self.layout_of(substs.type_at(0))?;
                // If ret_layout is unsigned, we checked that so is the distance, so we are good.
                let val = ImmTy::from_int(dist, ret_layout);
                let size = ImmTy::from_int(pointee_layout.size.bytes(), ret_layout);
                self.exact_div(&val, &size, dest)?;
            }

            sym::transmute => {
                self.copy_op(&args[0], dest, /*allow_transmute*/ true)?;
            }
            sym::assert_inhabited
            | sym::assert_zero_valid
            | sym::assert_mem_uninitialized_valid => {
                let ty = instance.substs.type_at(0);
                let layout = self.layout_of(ty)?;

                // For *all* intrinsics we first check `is_uninhabited` to give a more specific
                // error message.
                if layout.abi.is_uninhabited() {
                    // The run-time intrinsic panics just to get a good backtrace; here we abort
                    // since there is no problem showing a backtrace even for aborts.
                    M::abort(
                        self,
                        format!(
                            "aborted execution: attempted to instantiate uninhabited type `{}`",
                            ty
                        ),
                    )?;
                }

                if intrinsic_name == sym::assert_zero_valid {
                    let should_panic = !self
                        .tcx
                        .check_validity_of_init((InitKind::Zero, self.param_env.and(ty)))
                        .map_err(|_| err_inval!(TooGeneric))?;

                    if should_panic {
                        M::abort(
                            self,
                            format!(
                                "aborted execution: attempted to zero-initialize type `{}`, which is invalid",
                                ty
                            ),
                        )?;
                    }
                }

                if intrinsic_name == sym::assert_mem_uninitialized_valid {
                    let should_panic = !self
                        .tcx
                        .check_validity_of_init((
                            InitKind::UninitMitigated0x01Fill,
                            self.param_env.and(ty),
                        ))
                        .map_err(|_| err_inval!(TooGeneric))?;

                    if should_panic {
                        M::abort(
                            self,
                            format!(
                                "aborted execution: attempted to leave type `{}` uninitialized, which is invalid",
                                ty
                            ),
                        )?;
                    }
                }
            }
            sym::simd_insert => {
                let index = u64::from(self.read_scalar(&args[1])?.to_u32()?);
                let elem = &args[2];
                let (input, input_len) = self.operand_to_simd(&args[0])?;
                let (dest, dest_len) = self.place_to_simd(dest)?;
                assert_eq!(input_len, dest_len, "Return vector length must match input length");
                assert!(
                    index < dest_len,
                    "Index `{}` must be in bounds of vector with length {}`",
                    index,
                    dest_len
                );

                for i in 0..dest_len {
                    let place = self.mplace_index(&dest, i)?;
                    let value = if i == index {
                        elem.clone()
                    } else {
                        self.mplace_index(&input, i)?.into()
                    };
                    self.copy_op(&value, &place.into(), /*allow_transmute*/ false)?;
                }
            }
            sym::simd_extract => {
                let index = u64::from(self.read_scalar(&args[1])?.to_u32()?);
                let (input, input_len) = self.operand_to_simd(&args[0])?;
                assert!(
                    index < input_len,
                    "index `{}` must be in bounds of vector with length `{}`",
                    index,
                    input_len
                );
                self.copy_op(
                    &self.mplace_index(&input, index)?.into(),
                    dest,
                    /*allow_transmute*/ false,
                )?;
            }
            sym::likely | sym::unlikely | sym::black_box => {
                // These just return their argument
                self.copy_op(&args[0], dest, /*allow_transmute*/ false)?;
            }
            sym::raw_eq => {
                let result = self.raw_eq_intrinsic(&args[0], &args[1])?;
                self.write_scalar(result, dest)?;
            }

            sym::vtable_size => {
                let ptr = self.read_pointer(&args[0])?;
                let (size, _align) = self.get_vtable_size_and_align(ptr)?;
                self.write_scalar(Scalar::from_target_usize(size.bytes(), self), dest)?;
            }
            sym::vtable_align => {
                let ptr = self.read_pointer(&args[0])?;
                let (_size, align) = self.get_vtable_size_and_align(ptr)?;
                self.write_scalar(Scalar::from_target_usize(align.bytes(), self), dest)?;
            }

            _ => return Ok(false),
        }

        trace!("{:?}", self.dump_place(**dest));
        self.go_to_block(ret);
        Ok(true)
    }

    pub(super) fn emulate_nondiverging_intrinsic(
        &mut self,
        intrinsic: &NonDivergingIntrinsic<'tcx>,
    ) -> InterpResult<'tcx> {
        match intrinsic {
            NonDivergingIntrinsic::Assume(op) => {
                let op = self.eval_operand(op, None)?;
                let cond = self.read_scalar(&op)?.to_bool()?;
                if !cond {
                    throw_ub_format!("`assume` called with `false`");
                }
                Ok(())
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

    pub fn exact_div(
        &mut self,
        a: &ImmTy<'tcx, M::Provenance>,
        b: &ImmTy<'tcx, M::Provenance>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        // Performs an exact division, resulting in undefined behavior where
        // `x % y != 0` or `y == 0` or `x == T::MIN && y == -1`.
        // First, check x % y != 0 (or if that computation overflows).
        let (res, overflow, _ty) = self.overflowing_binary_op(BinOp::Rem, &a, &b)?;
        assert!(!overflow); // All overflow is UB, so this should never return on overflow.
        if res.assert_bits(a.layout.size) != 0 {
            throw_ub_format!("exact_div: {} cannot be divided by {} without remainder", a, b)
        }
        // `Rem` says this is all right, so we can let `Div` do its job.
        self.binop_ignore_overflow(BinOp::Div, &a, &b, dest)
    }

    pub fn saturating_arith(
        &self,
        mir_op: BinOp,
        l: &ImmTy<'tcx, M::Provenance>,
        r: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>> {
        assert!(matches!(mir_op, BinOp::Add | BinOp::Sub));
        let (val, overflowed, _ty) = self.overflowing_binary_op(mir_op, l, r)?;
        Ok(if overflowed {
            let size = l.layout.size;
            let num_bits = size.bits();
            if l.layout.abi.is_signed() {
                // For signed ints the saturated value depends on the sign of the first
                // term since the sign of the second term can be inferred from this and
                // the fact that the operation has overflowed (if either is 0 no
                // overflow can occur)
                let first_term: u128 = l.to_scalar().to_bits(l.layout.size)?;
                let first_term_positive = first_term & (1 << (num_bits - 1)) == 0;
                if first_term_positive {
                    // Negative overflow not possible since the positive first term
                    // can only increase an (in range) negative term for addition
                    // or corresponding negated positive term for subtraction
                    Scalar::from_int(size.signed_int_max(), size)
                } else {
                    // Positive overflow not possible for similar reason
                    // max negative
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
    /// allocation. For integer pointers, we consider each of them their own tiny allocation of size
    /// 0, so offset-by-0 (and only 0) is okay -- except that null cannot be offset by _any_ value.
    pub fn ptr_offset_inbounds(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        pointee_ty: Ty<'tcx>,
        offset_count: i64,
    ) -> InterpResult<'tcx, Pointer<Option<M::Provenance>>> {
        // We cannot overflow i64 as a type's size must be <= isize::MAX.
        let pointee_size = i64::try_from(self.layout_of(pointee_ty)?.size.bytes()).unwrap();
        // The computed offset, in bytes, must not overflow an isize.
        // `checked_mul` enforces a too small bound, but no actual allocation can be big enough for
        // the difference to be noticeable.
        let offset_bytes =
            offset_count.checked_mul(pointee_size).ok_or(err_ub!(PointerArithOverflow))?;
        // The offset being in bounds cannot rely on "wrapping around" the address space.
        // So, first rule out overflows in the pointer arithmetic.
        let offset_ptr = ptr.signed_offset(offset_bytes, self)?;
        // ptr and offset_ptr must be in bounds of the same allocated object. This means all of the
        // memory between these pointers must be accessible. Note that we do not require the
        // pointers to be properly aligned (unlike a read/write operation).
        let min_ptr = if offset_bytes >= 0 { ptr } else { offset_ptr };
        // This call handles checking for integer/null pointers.
        self.check_ptr_access_align(
            min_ptr,
            Size::from_bytes(offset_bytes.unsigned_abs()),
            Align::ONE,
            CheckInAllocMsg::PointerArithmeticTest,
        )?;
        Ok(offset_ptr)
    }

    /// Copy `count*size_of::<T>()` many bytes from `*src` to `*dst`.
    pub(crate) fn copy_intrinsic(
        &mut self,
        src: &OpTy<'tcx, <M as Machine<'mir, 'tcx>>::Provenance>,
        dst: &OpTy<'tcx, <M as Machine<'mir, 'tcx>>::Provenance>,
        count: &OpTy<'tcx, <M as Machine<'mir, 'tcx>>::Provenance>,
        nonoverlapping: bool,
    ) -> InterpResult<'tcx> {
        let count = self.read_target_usize(&count)?;
        let layout = self.layout_of(src.layout.ty.builtin_deref(true).unwrap().ty)?;
        let (size, align) = (layout.size, layout.align.abi);
        // `checked_mul` enforces a too small bound (the correct one would probably be target_isize_max),
        // but no actual allocation can be big enough for the difference to be noticeable.
        let size = size.checked_mul(count, self).ok_or_else(|| {
            err_ub_format!(
                "overflow computing total size of `{}`",
                if nonoverlapping { "copy_nonoverlapping" } else { "copy" }
            )
        })?;

        let src = self.read_pointer(&src)?;
        let dst = self.read_pointer(&dst)?;

        self.mem_copy(src, align, dst, align, size, nonoverlapping)
    }

    pub(crate) fn write_bytes_intrinsic(
        &mut self,
        dst: &OpTy<'tcx, <M as Machine<'mir, 'tcx>>::Provenance>,
        byte: &OpTy<'tcx, <M as Machine<'mir, 'tcx>>::Provenance>,
        count: &OpTy<'tcx, <M as Machine<'mir, 'tcx>>::Provenance>,
    ) -> InterpResult<'tcx> {
        let layout = self.layout_of(dst.layout.ty.builtin_deref(true).unwrap().ty)?;

        let dst = self.read_pointer(&dst)?;
        let byte = self.read_scalar(&byte)?.to_u8()?;
        let count = self.read_target_usize(&count)?;

        // `checked_mul` enforces a too small bound (the correct one would probably be target_isize_max),
        // but no actual allocation can be big enough for the difference to be noticeable.
        let len = layout
            .size
            .checked_mul(count, self)
            .ok_or_else(|| err_ub_format!("overflow computing total size of `write_bytes`"))?;

        let bytes = std::iter::repeat(byte).take(len.bytes_usize());
        self.write_bytes_ptr(dst, bytes)
    }

    pub(crate) fn raw_eq_intrinsic(
        &mut self,
        lhs: &OpTy<'tcx, <M as Machine<'mir, 'tcx>>::Provenance>,
        rhs: &OpTy<'tcx, <M as Machine<'mir, 'tcx>>::Provenance>,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>> {
        let layout = self.layout_of(lhs.layout.ty.builtin_deref(true).unwrap().ty)?;
        assert!(layout.is_sized());

        let get_bytes = |this: &InterpCx<'mir, 'tcx, M>,
                         op: &OpTy<'tcx, <M as Machine<'mir, 'tcx>>::Provenance>,
                         size|
         -> InterpResult<'tcx, &[u8]> {
            let ptr = this.read_pointer(op)?;
            let Some(alloc_ref) = self.get_ptr_alloc(ptr, size, Align::ONE)? else {
                // zero-sized access
                return Ok(&[]);
            };
            if alloc_ref.has_provenance() {
                throw_ub_format!("`raw_eq` on bytes with provenance");
            }
            alloc_ref.get_bytes_strip_provenance()
        };

        let lhs_bytes = get_bytes(self, lhs, layout.size)?;
        let rhs_bytes = get_bytes(self, rhs, layout.size)?;
        Ok(Scalar::from_bool(lhs_bytes == rhs_bytes))
    }
}
