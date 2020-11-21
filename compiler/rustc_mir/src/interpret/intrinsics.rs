//! Intrinsics and other functions that the miri engine executes without
//! looking at their MIR. Intrinsics/functions supported here are shared by CTFE
//! and miri.

use std::convert::TryFrom;

use rustc_hir::def_id::DefId;
use rustc_middle::mir::{
    self,
    interpret::{uabs, ConstValue, GlobalId, InterpResult, Scalar},
    BinOp,
};
use rustc_middle::ty;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_span::symbol::{sym, Symbol};
use rustc_target::abi::{Abi, LayoutOf as _, Primitive, Size};

use super::{
    util::ensure_monomorphic_enough, CheckInAllocMsg, ImmTy, InterpCx, Machine, OpTy, PlaceTy,
};

mod caller_location;
mod type_name;

fn numeric_intrinsic<'tcx, Tag>(
    name: Symbol,
    bits: u128,
    kind: Primitive,
) -> InterpResult<'tcx, Scalar<Tag>> {
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
    Ok(Scalar::from_uint(bits_out, size))
}

/// The logic for all nullary intrinsics is implemented here. These intrinsics don't get evaluated
/// inside an `InterpCx` and instead have their value computed directly from rustc internal info.
crate fn eval_nullary_intrinsic<'tcx>(
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
            let alloc = type_name::alloc_type_name(tcx, tp_ty);
            ConstValue::Slice { data: alloc, start: 0, end: alloc.len() }
        }
        sym::needs_drop => ConstValue::from_bool(tp_ty.needs_drop(tcx, param_env)),
        sym::size_of | sym::min_align_of | sym::pref_align_of => {
            let layout = tcx.layout_of(param_env.and(tp_ty)).map_err(|e| err_inval!(Layout(e)))?;
            let n = match name {
                sym::pref_align_of => layout.align.pref.bytes(),
                sym::min_align_of => layout.align.abi.bytes(),
                sym::size_of => layout.size.bytes(),
                _ => bug!(),
            };
            ConstValue::from_machine_usize(n, &tcx)
        }
        sym::type_id => {
            ensure_monomorphic_enough(tcx, tp_ty)?;
            ConstValue::from_u64(tcx.type_id_hash(tp_ty))
        }
        sym::variant_count => match tp_ty.kind() {
            ty::Adt(ref adt, _) => ConstValue::from_machine_usize(adt.variants.len() as u64, &tcx),
            ty::Projection(_)
            | ty::Opaque(_, _)
            | ty::Param(_)
            | ty::Bound(_, _)
            | ty::Placeholder(_)
            | ty::Infer(_) => throw_inval!(TooGeneric),
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
            | ty::Dynamic(_, _)
            | ty::Closure(_, _)
            | ty::Generator(_, _, _)
            | ty::GeneratorWitness(_)
            | ty::Never
            | ty::Tuple(_)
            | ty::Error(_) => ConstValue::from_machine_usize(0u64, &tcx),
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
        args: &[OpTy<'tcx, M::PointerTag>],
        ret: Option<(PlaceTy<'tcx, M::PointerTag>, mir::BasicBlock)>,
    ) -> InterpResult<'tcx, bool> {
        let substs = instance.substs;
        let intrinsic_name = self.tcx.item_name(instance.def_id());

        // First handle intrinsics without return place.
        let (dest, ret) = match ret {
            None => match intrinsic_name {
                sym::transmute => throw_ub_format!("transmuting to uninhabited type"),
                sym::unreachable => throw_ub!(Unreachable),
                sym::abort => M::abort(self)?,
                // Unsupported diverging intrinsic.
                _ => return Ok(false),
            },
            Some(p) => p,
        };

        // Keep the patterns in this match ordered the same as the list in
        // `src/librustc_middle/ty/constness.rs`
        match intrinsic_name {
            sym::caller_location => {
                let span = self.find_closest_untracked_caller_location();
                let location = self.alloc_caller_location_for_span(span);
                self.write_scalar(location.ptr, dest)?;
            }

            sym::min_align_of_val | sym::size_of_val => {
                let place = self.deref_operand(args[0])?;
                let (size, align) = self
                    .size_and_align_of(place.meta, place.layout)?
                    .ok_or_else(|| err_unsup_format!("`extern type` does not have known layout"))?;

                let result = match intrinsic_name {
                    sym::min_align_of_val => align.bytes(),
                    sym::size_of_val => size.bytes(),
                    _ => bug!(),
                };

                self.write_scalar(Scalar::from_machine_usize(result, self), dest)?;
            }

            sym::min_align_of
            | sym::pref_align_of
            | sym::needs_drop
            | sym::size_of
            | sym::type_id
            | sym::type_name
            | sym::variant_count => {
                let gid = GlobalId { instance, promoted: None };
                let ty = match intrinsic_name {
                    sym::min_align_of | sym::pref_align_of | sym::size_of | sym::variant_count => {
                        self.tcx.types.usize
                    }
                    sym::needs_drop => self.tcx.types.bool,
                    sym::type_id => self.tcx.types.u64,
                    sym::type_name => self.tcx.mk_static_str(),
                    _ => bug!("already checked for nullary intrinsics"),
                };
                let val =
                    self.tcx.const_eval_global_id(self.param_env, gid, Some(self.tcx.span))?;
                let const_ = ty::Const { val: ty::ConstKind::Value(val), ty };
                let val = self.const_to_op(&const_, None)?;
                self.copy_op(val, dest)?;
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
                let val = self.read_scalar(args[0])?.check_init()?;
                let bits = self.force_bits(val, layout_of.size)?;
                let kind = match layout_of.abi {
                    Abi::Scalar(ref scalar) => scalar.value,
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
                let out_val = numeric_intrinsic(intrinsic_name, bits, kind)?;
                self.write_scalar(out_val, dest)?;
            }
            sym::wrapping_add
            | sym::wrapping_sub
            | sym::wrapping_mul
            | sym::add_with_overflow
            | sym::sub_with_overflow
            | sym::mul_with_overflow => {
                let lhs = self.read_immediate(args[0])?;
                let rhs = self.read_immediate(args[1])?;
                let (bin_op, ignore_overflow) = match intrinsic_name {
                    sym::wrapping_add => (BinOp::Add, true),
                    sym::wrapping_sub => (BinOp::Sub, true),
                    sym::wrapping_mul => (BinOp::Mul, true),
                    sym::add_with_overflow => (BinOp::Add, false),
                    sym::sub_with_overflow => (BinOp::Sub, false),
                    sym::mul_with_overflow => (BinOp::Mul, false),
                    _ => bug!("Already checked for int ops"),
                };
                if ignore_overflow {
                    self.binop_ignore_overflow(bin_op, lhs, rhs, dest)?;
                } else {
                    self.binop_with_overflow(bin_op, lhs, rhs, dest)?;
                }
            }
            sym::saturating_add | sym::saturating_sub => {
                let l = self.read_immediate(args[0])?;
                let r = self.read_immediate(args[1])?;
                let is_add = intrinsic_name == sym::saturating_add;
                let (val, overflowed, _ty) =
                    self.overflowing_binary_op(if is_add { BinOp::Add } else { BinOp::Sub }, l, r)?;
                let val = if overflowed {
                    let num_bits = l.layout.size.bits();
                    if l.layout.abi.is_signed() {
                        // For signed ints the saturated value depends on the sign of the first
                        // term since the sign of the second term can be inferred from this and
                        // the fact that the operation has overflowed (if either is 0 no
                        // overflow can occur)
                        let first_term: u128 = self.force_bits(l.to_scalar()?, l.layout.size)?;
                        let first_term_positive = first_term & (1 << (num_bits - 1)) == 0;
                        if first_term_positive {
                            // Negative overflow not possible since the positive first term
                            // can only increase an (in range) negative term for addition
                            // or corresponding negated positive term for subtraction
                            Scalar::from_uint(
                                (1u128 << (num_bits - 1)) - 1, // max positive
                                Size::from_bits(num_bits),
                            )
                        } else {
                            // Positive overflow not possible for similar reason
                            // max negative
                            Scalar::from_uint(1u128 << (num_bits - 1), Size::from_bits(num_bits))
                        }
                    } else {
                        // unsigned
                        if is_add {
                            // max unsigned
                            Scalar::from_uint(
                                u128::MAX >> (128 - num_bits),
                                Size::from_bits(num_bits),
                            )
                        } else {
                            // underflow to 0
                            Scalar::from_uint(0u128, Size::from_bits(num_bits))
                        }
                    }
                } else {
                    val
                };
                self.write_scalar(val, dest)?;
            }
            sym::discriminant_value => {
                let place = self.deref_operand(args[0])?;
                let discr_val = self.read_discriminant(place.into())?.0;
                self.write_scalar(discr_val, dest)?;
            }
            sym::unchecked_shl
            | sym::unchecked_shr
            | sym::unchecked_add
            | sym::unchecked_sub
            | sym::unchecked_mul
            | sym::unchecked_div
            | sym::unchecked_rem => {
                let l = self.read_immediate(args[0])?;
                let r = self.read_immediate(args[1])?;
                let bin_op = match intrinsic_name {
                    sym::unchecked_shl => BinOp::Shl,
                    sym::unchecked_shr => BinOp::Shr,
                    sym::unchecked_add => BinOp::Add,
                    sym::unchecked_sub => BinOp::Sub,
                    sym::unchecked_mul => BinOp::Mul,
                    sym::unchecked_div => BinOp::Div,
                    sym::unchecked_rem => BinOp::Rem,
                    _ => bug!("Already checked for int ops"),
                };
                let (val, overflowed, _ty) = self.overflowing_binary_op(bin_op, l, r)?;
                if overflowed {
                    let layout = self.layout_of(substs.type_at(0))?;
                    let r_val = self.force_bits(r.to_scalar()?, layout.size)?;
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
                let val = self.read_scalar(args[0])?.check_init()?;
                let val_bits = self.force_bits(val, layout.size)?;
                let raw_shift = self.read_scalar(args[1])?.check_init()?;
                let raw_shift_bits = self.force_bits(raw_shift, layout.size)?;
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
            sym::offset => {
                let ptr = self.read_scalar(args[0])?.check_init()?;
                let offset_count = self.read_scalar(args[1])?.to_machine_isize(self)?;
                let pointee_ty = substs.type_at(0);

                let offset_ptr = self.ptr_offset_inbounds(ptr, pointee_ty, offset_count)?;
                self.write_scalar(offset_ptr, dest)?;
            }
            sym::arith_offset => {
                let ptr = self.read_scalar(args[0])?.check_init()?;
                let offset_count = self.read_scalar(args[1])?.to_machine_isize(self)?;
                let pointee_ty = substs.type_at(0);

                let pointee_size = i64::try_from(self.layout_of(pointee_ty)?.size.bytes()).unwrap();
                let offset_bytes = offset_count.wrapping_mul(pointee_size);
                let offset_ptr = ptr.ptr_wrapping_signed_offset(offset_bytes, self);
                self.write_scalar(offset_ptr, dest)?;
            }
            sym::ptr_offset_from => {
                let a = self.read_immediate(args[0])?.to_scalar()?;
                let b = self.read_immediate(args[1])?.to_scalar()?;

                // Special case: if both scalars are *equal integers*
                // and not NULL, we pretend there is an allocation of size 0 right there,
                // and their offset is 0. (There's never a valid object at NULL, making it an
                // exception from the exception.)
                // This is the dual to the special exception for offset-by-0
                // in the inbounds pointer offset operation (see the Miri code, `src/operator.rs`).
                //
                // Control flow is weird because we cannot early-return (to reach the
                // `go_to_block` at the end).
                let done = if a.is_bits() && b.is_bits() {
                    let a = a.to_machine_usize(self)?;
                    let b = b.to_machine_usize(self)?;
                    if a == b && a != 0 {
                        self.write_scalar(Scalar::from_machine_isize(0, self), dest)?;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                if !done {
                    // General case: we need two pointers.
                    let a = self.force_ptr(a)?;
                    let b = self.force_ptr(b)?;
                    if a.alloc_id != b.alloc_id {
                        throw_ub_format!(
                            "ptr_offset_from cannot compute offset of pointers into different \
                            allocations.",
                        );
                    }
                    let usize_layout = self.layout_of(self.tcx.types.usize)?;
                    let isize_layout = self.layout_of(self.tcx.types.isize)?;
                    let a_offset = ImmTy::from_uint(a.offset.bytes(), usize_layout);
                    let b_offset = ImmTy::from_uint(b.offset.bytes(), usize_layout);
                    let (val, _overflowed, _ty) =
                        self.overflowing_binary_op(BinOp::Sub, a_offset, b_offset)?;
                    let pointee_layout = self.layout_of(substs.type_at(0))?;
                    let val = ImmTy::from_scalar(val, isize_layout);
                    let size = ImmTy::from_int(pointee_layout.size.bytes(), isize_layout);
                    self.exact_div(val, size, dest)?;
                }
            }

            sym::transmute => {
                self.copy_op_transmute(args[0], dest)?;
            }
            sym::simd_insert => {
                let index = u64::from(self.read_scalar(args[1])?.to_u32()?);
                let elem = args[2];
                let input = args[0];
                let (len, e_ty) = input.layout.ty.simd_size_and_type(*self.tcx);
                assert!(
                    index < len,
                    "Index `{}` must be in bounds of vector type `{}`: `[0, {})`",
                    index,
                    e_ty,
                    len
                );
                assert_eq!(
                    input.layout, dest.layout,
                    "Return type `{}` must match vector type `{}`",
                    dest.layout.ty, input.layout.ty
                );
                assert_eq!(
                    elem.layout.ty, e_ty,
                    "Scalar element type `{}` must match vector element type `{}`",
                    elem.layout.ty, e_ty
                );

                for i in 0..len {
                    let place = self.place_index(dest, i)?;
                    let value = if i == index { elem } else { self.operand_index(input, i)? };
                    self.copy_op(value, place)?;
                }
            }
            sym::simd_extract => {
                let index = u64::from(self.read_scalar(args[1])?.to_u32()?);
                let (len, e_ty) = args[0].layout.ty.simd_size_and_type(*self.tcx);
                assert!(
                    index < len,
                    "index `{}` is out-of-bounds of vector type `{}` with length `{}`",
                    index,
                    e_ty,
                    len
                );
                assert_eq!(
                    e_ty, dest.layout.ty,
                    "Return type `{}` must match vector element type `{}`",
                    dest.layout.ty, e_ty
                );
                self.copy_op(self.operand_index(args[0], index)?, dest)?;
            }
            sym::likely | sym::unlikely => {
                // These just return their argument
                self.copy_op(args[0], dest)?;
            }
            sym::assume => {
                let cond = self.read_scalar(args[0])?.check_init()?.to_bool()?;
                if !cond {
                    throw_ub_format!("`assume` intrinsic called with `false`");
                }
            }
            _ => return Ok(false),
        }

        trace!("{:?}", self.dump_place(*dest));
        self.go_to_block(ret);
        Ok(true)
    }

    pub fn exact_div(
        &mut self,
        a: ImmTy<'tcx, M::PointerTag>,
        b: ImmTy<'tcx, M::PointerTag>,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx> {
        // Performs an exact division, resulting in undefined behavior where
        // `x % y != 0` or `y == 0` or `x == T::MIN && y == -1`.
        // First, check x % y != 0 (or if that computation overflows).
        let (res, overflow, _ty) = self.overflowing_binary_op(BinOp::Rem, a, b)?;
        if overflow || res.assert_bits(a.layout.size) != 0 {
            // Then, check if `b` is -1, which is the "MIN / -1" case.
            let minus1 = Scalar::from_int(-1, dest.layout.size);
            let b_scalar = b.to_scalar().unwrap();
            if b_scalar == minus1 {
                throw_ub_format!("exact_div: result of dividing MIN by -1 cannot be represented")
            } else {
                throw_ub_format!("exact_div: {} cannot be divided by {} without remainder", a, b,)
            }
        }
        // `Rem` says this is all right, so we can let `Div` do its job.
        self.binop_ignore_overflow(BinOp::Div, a, b, dest)
    }

    /// Offsets a pointer by some multiple of its type, returning an error if the pointer leaves its
    /// allocation. For integer pointers, we consider each of them their own tiny allocation of size
    /// 0, so offset-by-0 (and only 0) is okay -- except that NULL cannot be offset by _any_ value.
    pub fn ptr_offset_inbounds(
        &self,
        ptr: Scalar<M::PointerTag>,
        pointee_ty: Ty<'tcx>,
        offset_count: i64,
    ) -> InterpResult<'tcx, Scalar<M::PointerTag>> {
        // We cannot overflow i64 as a type's size must be <= isize::MAX.
        let pointee_size = i64::try_from(self.layout_of(pointee_ty)?.size.bytes()).unwrap();
        // The computed offset, in bytes, cannot overflow an isize.
        let offset_bytes =
            offset_count.checked_mul(pointee_size).ok_or(err_ub!(PointerArithOverflow))?;
        // The offset being in bounds cannot rely on "wrapping around" the address space.
        // So, first rule out overflows in the pointer arithmetic.
        let offset_ptr = ptr.ptr_signed_offset(offset_bytes, self)?;
        // ptr and offset_ptr must be in bounds of the same allocated object. This means all of the
        // memory between these pointers must be accessible. Note that we do not require the
        // pointers to be properly aligned (unlike a read/write operation).
        let min_ptr = if offset_bytes >= 0 { ptr } else { offset_ptr };
        let size: u64 = uabs(offset_bytes);
        // This call handles checking for integer/NULL pointers.
        self.memory.check_ptr_access_align(
            min_ptr,
            Size::from_bytes(size),
            None,
            CheckInAllocMsg::InboundsTest,
        )?;
        Ok(offset_ptr)
    }
}
