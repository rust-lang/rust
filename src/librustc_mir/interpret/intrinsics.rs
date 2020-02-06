//! Intrinsics and other functions that the miri engine executes without
//! looking at their MIR. Intrinsics/functions supported here are shared by CTFE
//! and miri.

use rustc::mir::{
    self,
    interpret::{ConstValue, GlobalId, InterpResult, Scalar},
    BinOp,
};
use rustc::ty;
use rustc::ty::layout::{LayoutOf, Primitive, Size};
use rustc::ty::subst::SubstsRef;
use rustc::ty::TyCtxt;
use rustc_hir::def_id::DefId;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;

use super::{ImmTy, InterpCx, Machine, OpTy, PlaceTy};

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
    let extra = 128 - size.bits() as u128;
    let bits_out = match name {
        sym::ctpop => bits.count_ones() as u128,
        sym::ctlz => bits.leading_zeros() as u128 - extra,
        sym::cttz => (bits << extra).trailing_zeros() as u128 - extra,
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
) -> InterpResult<'tcx, &'tcx ty::Const<'tcx>> {
    let tp_ty = substs.type_at(0);
    let name = tcx.item_name(def_id);
    Ok(match name {
        sym::type_name => {
            let alloc = type_name::alloc_type_name(tcx, tp_ty);
            tcx.mk_const(ty::Const {
                val: ty::ConstKind::Value(ConstValue::Slice {
                    data: alloc,
                    start: 0,
                    end: alloc.len(),
                }),
                ty: tcx.mk_static_str(),
            })
        }
        sym::needs_drop => ty::Const::from_bool(tcx, tp_ty.needs_drop(tcx, param_env)),
        sym::size_of | sym::min_align_of | sym::pref_align_of => {
            let layout = tcx.layout_of(param_env.and(tp_ty)).map_err(|e| err_inval!(Layout(e)))?;
            let n = match name {
                sym::pref_align_of => layout.align.pref.bytes(),
                sym::min_align_of => layout.align.abi.bytes(),
                sym::size_of => layout.size.bytes(),
                _ => bug!(),
            };
            ty::Const::from_usize(tcx, n)
        }
        sym::type_id => {
            ty::Const::from_bits(tcx, tcx.type_id_hash(tp_ty).into(), param_env.and(tcx.types.u64))
        }
        other => bug!("`{}` is not a zero arg intrinsic", other),
    })
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Returns `true` if emulation happened.
    pub fn emulate_intrinsic(
        &mut self,
        span: Span,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, M::PointerTag>],
        ret: Option<(PlaceTy<'tcx, M::PointerTag>, mir::BasicBlock)>,
    ) -> InterpResult<'tcx, bool> {
        let substs = instance.substs;
        let intrinsic_name = self.tcx.item_name(instance.def_id());

        // We currently do not handle any intrinsics that are *allowed* to diverge,
        // but `transmute` could lack a return place in case of UB.
        let (dest, ret) = match ret {
            Some(p) => p,
            None => match intrinsic_name {
                sym::transmute => throw_ub!(Unreachable),
                _ => return Ok(false),
            },
        };

        // Keep the patterns in this match ordered the same as the list in
        // `src/librustc/ty/constness.rs`
        match intrinsic_name {
            sym::caller_location => {
                let span = self.find_closest_untracked_caller_location().unwrap_or(span);
                let location = self.alloc_caller_location_for_span(span);
                self.write_scalar(location.ptr, dest)?;
            }

            sym::min_align_of
            | sym::pref_align_of
            | sym::needs_drop
            | sym::size_of
            | sym::type_id
            | sym::type_name => {
                let gid = GlobalId { instance, promoted: None };
                let val = self.const_eval(gid)?;
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
                let val = self.read_scalar(args[0])?.not_undef()?;
                let bits = self.force_bits(val, layout_of.size)?;
                let kind = match layout_of.abi {
                    ty::layout::Abi::Scalar(ref scalar) => scalar.value,
                    _ => throw_unsup!(TypeNotPrimitive(ty)),
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
                                u128::max_value() >> (128 - num_bits),
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
                        throw_ub_format!("Overflowing shift by {} in `{}`", r_val, intrinsic_name);
                    } else {
                        throw_ub_format!("Overflow executing `{}`", intrinsic_name);
                    }
                }
                self.write_scalar(val, dest)?;
            }
            sym::rotate_left | sym::rotate_right => {
                // rotate_left: (X << (S % BW)) | (X >> ((BW - S) % BW))
                // rotate_right: (X << ((BW - S) % BW)) | (X >> (S % BW))
                let layout = self.layout_of(substs.type_at(0))?;
                let val = self.read_scalar(args[0])?.not_undef()?;
                let val_bits = self.force_bits(val, layout.size)?;
                let raw_shift = self.read_scalar(args[1])?.not_undef()?;
                let raw_shift_bits = self.force_bits(raw_shift, layout.size)?;
                let width_bits = layout.size.bits() as u128;
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

            sym::ptr_offset_from => {
                let isize_layout = self.layout_of(self.tcx.types.isize)?;
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
                        self.write_scalar(Scalar::from_int(0, isize_layout.size), dest)?;
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
                let (len, e_ty) = input.layout.ty.simd_size_and_type(self.tcx.tcx);
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
                    let place = self.place_field(dest, i)?;
                    let value = if i == index { elem } else { self.operand_field(input, i)? };
                    self.copy_op(value, place)?;
                }
            }
            sym::simd_extract => {
                let index = u64::from(self.read_scalar(args[1])?.to_u32()?);
                let (len, e_ty) = args[0].layout.ty.simd_size_and_type(self.tcx.tcx);
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
                self.copy_op(self.operand_field(args[0], index)?, dest)?;
            }
            _ => return Ok(false),
        }

        self.dump_place(*dest);
        self.go_to_block(ret);
        Ok(true)
    }

    /// "Intercept" a function call to a panic-related function
    /// because we have something special to do for it.
    /// Returns `true` if an intercept happened.
    pub fn hook_panic_fn(
        &mut self,
        span: Span,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, M::PointerTag>],
    ) -> InterpResult<'tcx, bool> {
        let def_id = instance.def_id();
        if Some(def_id) == self.tcx.lang_items().panic_fn()
            || Some(def_id) == self.tcx.lang_items().begin_panic_fn()
        {
            // &'static str
            assert!(args.len() == 1);

            let msg_place = self.deref_operand(args[0])?;
            let msg = Symbol::intern(self.read_str(msg_place)?);
            let span = self.find_closest_untracked_caller_location().unwrap_or(span);
            let (file, line, col) = self.location_triple_for_span(span);
            throw_panic!(Panic { msg, file, line, col })
        } else {
            return Ok(false);
        }
    }

    pub fn exact_div(
        &mut self,
        a: ImmTy<'tcx, M::PointerTag>,
        b: ImmTy<'tcx, M::PointerTag>,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx> {
        // Performs an exact division, resulting in undefined behavior where
        // `x % y != 0` or `y == 0` or `x == T::min_value() && y == -1`.
        // First, check x % y != 0.
        if self.binary_op(BinOp::Rem, a, b)?.to_bits()? != 0 {
            // Then, check if `b` is -1, which is the "min_value / -1" case.
            let minus1 = Scalar::from_int(-1, dest.layout.size);
            let b_scalar = b.to_scalar().unwrap();
            if b_scalar == minus1 {
                throw_ub_format!("exact_div: result of dividing MIN by -1 cannot be represented")
            } else {
                throw_ub_format!("exact_div: {} cannot be divided by {} without remainder", a, b,)
            }
        }
        self.binop_ignore_overflow(BinOp::Div, a, b, dest)
    }
}
