//! Intrinsics and other functions that the miri engine executes without
//! looking at their MIR. Intrinsics/functions supported here are shared by CTFE
//! and miri.

use syntax::symbol::Symbol;
use syntax_pos::Span;
use rustc::ty;
use rustc::ty::layout::{LayoutOf, Primitive, Size};
use rustc::ty::subst::SubstsRef;
use rustc::hir::def_id::DefId;
use rustc::ty::TyCtxt;
use rustc::mir::{
    self, BinOp,
    interpret::{InterpResult, Scalar, GlobalId, ConstValue}
};

use super::{
    Machine, PlaceTy, OpTy, InterpCx, ImmTy,
};

mod caller_location;
mod type_name;

fn numeric_intrinsic<'tcx, Tag>(
    name: &str,
    bits: u128,
    kind: Primitive,
) -> InterpResult<'tcx, Scalar<Tag>> {
    let size = match kind {
        Primitive::Int(integer, _) => integer.size(),
        _ => bug!("invalid `{}` argument: {:?}", name, bits),
    };
    let extra = 128 - size.bits() as u128;
    let bits_out = match name {
        "ctpop" => bits.count_ones() as u128,
        "ctlz" => bits.leading_zeros() as u128 - extra,
        "cttz" => (bits << extra).trailing_zeros() as u128 - extra,
        "bswap" => (bits << extra).swap_bytes(),
        "bitreverse" => (bits << extra).reverse_bits(),
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
    let name = &*tcx.item_name(def_id).as_str();
    Ok(match name {
        "type_name" => {
            let alloc = type_name::alloc_type_name(tcx, tp_ty);
            tcx.mk_const(ty::Const {
                val: ty::ConstKind::Value(ConstValue::Slice {
                    data: alloc,
                    start: 0,
                    end: alloc.len(),
                }),
                ty: tcx.mk_static_str(),
            })
        },
        "needs_drop" => ty::Const::from_bool(tcx, tp_ty.needs_drop(tcx, param_env)),
        "size_of" |
        "min_align_of" |
        "pref_align_of" => {
            let layout = tcx.layout_of(param_env.and(tp_ty)).map_err(|e| err_inval!(Layout(e)))?;
            let n = match name {
                "pref_align_of" => layout.align.pref.bytes(),
                "min_align_of" => layout.align.abi.bytes(),
                "size_of" => layout.size.bytes(),
                _ => bug!(),
            };
            ty::Const::from_usize(tcx, n)
        },
        "type_id" => ty::Const::from_bits(
            tcx,
            tcx.type_id_hash(tp_ty).into(),
            param_env.and(tcx.types.u64),
        ),
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
        let intrinsic_name = &*self.tcx.item_name(instance.def_id()).as_str();

        // We currently do not handle any intrinsics that are *allowed* to diverge,
        // but `transmute` could lack a return place in case of UB.
        let (dest, ret) = match ret {
            Some(p) => p,
            None => match intrinsic_name {
                "transmute" => throw_ub!(Unreachable),
                _ => return Ok(false),
            }
        };

        match intrinsic_name {
            "caller_location" => {
                let topmost = span.ctxt().outer_expn().expansion_cause().unwrap_or(span);
                let caller = self.tcx.sess.source_map().lookup_char_pos(topmost.lo());
                let location = self.alloc_caller_location(
                    Symbol::intern(&caller.file.name.to_string()),
                    caller.line as u32,
                    caller.col_display as u32 + 1,
                )?;
                self.write_scalar(location.ptr, dest)?;
            }

            "min_align_of" |
            "pref_align_of" |
            "needs_drop" |
            "size_of" |
            "type_id" |
            "type_name" => {
                let gid = GlobalId {
                    instance,
                    promoted: None,
                };
                let val = self.tcx.const_eval(self.param_env.and(gid))?;
                let val = self.eval_const_to_op(val, None)?;
                self.copy_op(val, dest)?;
            }

            | "ctpop"
            | "cttz"
            | "cttz_nonzero"
            | "ctlz"
            | "ctlz_nonzero"
            | "bswap"
            | "bitreverse" => {
                let ty = substs.type_at(0);
                let layout_of = self.layout_of(ty)?;
                let val = self.read_scalar(args[0])?.not_undef()?;
                let bits = self.force_bits(val, layout_of.size)?;
                let kind = match layout_of.abi {
                    ty::layout::Abi::Scalar(ref scalar) => scalar.value,
                    _ => throw_unsup!(TypeNotPrimitive(ty)),
                };
                let out_val = if intrinsic_name.ends_with("_nonzero") {
                    if bits == 0 {
                        throw_ub_format!("`{}` called on 0", intrinsic_name);
                    }
                    numeric_intrinsic(intrinsic_name.trim_end_matches("_nonzero"), bits, kind)?
                } else {
                    numeric_intrinsic(intrinsic_name, bits, kind)?
                };
                self.write_scalar(out_val, dest)?;
            }
            | "wrapping_add"
            | "wrapping_sub"
            | "wrapping_mul"
            | "add_with_overflow"
            | "sub_with_overflow"
            | "mul_with_overflow" => {
                let lhs = self.read_immediate(args[0])?;
                let rhs = self.read_immediate(args[1])?;
                let (bin_op, ignore_overflow) = match intrinsic_name {
                    "wrapping_add" => (BinOp::Add, true),
                    "wrapping_sub" => (BinOp::Sub, true),
                    "wrapping_mul" => (BinOp::Mul, true),
                    "add_with_overflow" => (BinOp::Add, false),
                    "sub_with_overflow" => (BinOp::Sub, false),
                    "mul_with_overflow" => (BinOp::Mul, false),
                    _ => bug!("Already checked for int ops")
                };
                if ignore_overflow {
                    self.binop_ignore_overflow(bin_op, lhs, rhs, dest)?;
                } else {
                    self.binop_with_overflow(bin_op, lhs, rhs, dest)?;
                }
            }
            "saturating_add" | "saturating_sub" => {
                let l = self.read_immediate(args[0])?;
                let r = self.read_immediate(args[1])?;
                let is_add = intrinsic_name == "saturating_add";
                let (val, overflowed, _ty) = self.overflowing_binary_op(if is_add {
                    BinOp::Add
                } else {
                    BinOp::Sub
                }, l, r)?;
                let val = if overflowed {
                    let num_bits = l.layout.size.bits();
                    if l.layout.abi.is_signed() {
                        // For signed ints the saturated value depends on the sign of the first
                        // term since the sign of the second term can be inferred from this and
                        // the fact that the operation has overflowed (if either is 0 no
                        // overflow can occur)
                        let first_term: u128 = self.force_bits(l.to_scalar()?, l.layout.size)?;
                        let first_term_positive = first_term & (1 << (num_bits-1)) == 0;
                        if first_term_positive {
                            // Negative overflow not possible since the positive first term
                            // can only increase an (in range) negative term for addition
                            // or corresponding negated positive term for subtraction
                            Scalar::from_uint((1u128 << (num_bits - 1)) - 1,  // max positive
                                Size::from_bits(num_bits))
                        } else {
                            // Positive overflow not possible for similar reason
                            // max negative
                            Scalar::from_uint(1u128 << (num_bits - 1), Size::from_bits(num_bits))
                        }
                    } else {  // unsigned
                        if is_add {
                            // max unsigned
                            Scalar::from_uint(u128::max_value() >> (128 - num_bits),
                                Size::from_bits(num_bits))
                        } else {  // underflow to 0
                            Scalar::from_uint(0u128, Size::from_bits(num_bits))
                        }
                    }
                } else {
                    val
                };
                self.write_scalar(val, dest)?;
            }
            "unchecked_shl" | "unchecked_shr" => {
                let l = self.read_immediate(args[0])?;
                let r = self.read_immediate(args[1])?;
                let bin_op = match intrinsic_name {
                    "unchecked_shl" => BinOp::Shl,
                    "unchecked_shr" => BinOp::Shr,
                    _ => bug!("Already checked for int ops")
                };
                let (val, overflowed, _ty) = self.overflowing_binary_op(bin_op, l, r)?;
                if overflowed {
                    let layout = self.layout_of(substs.type_at(0))?;
                    let r_val = self.force_bits(r.to_scalar()?, layout.size)?;
                    throw_ub_format!("Overflowing shift by {} in `{}`", r_val, intrinsic_name);
                }
                self.write_scalar(val, dest)?;
            }
            "rotate_left" | "rotate_right" => {
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
                let result_bits = if intrinsic_name == "rotate_left" {
                    (val_bits << shift_bits) | (val_bits >> inv_shift_bits)
                } else {
                    (val_bits >> shift_bits) | (val_bits << inv_shift_bits)
                };
                let truncated_bits = self.truncate(result_bits, layout);
                let result = Scalar::from_uint(truncated_bits, layout.size);
                self.write_scalar(result, dest)?;
            }

            "ptr_offset_from" => {
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
                    } else { false }
                } else { false };

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
                    let (val, _overflowed, _ty) = self.overflowing_binary_op(
                        BinOp::Sub, a_offset, b_offset,
                    )?;
                    let pointee_layout = self.layout_of(substs.type_at(0))?;
                    let val = ImmTy::from_scalar(val, isize_layout);
                    let size = ImmTy::from_int(pointee_layout.size.bytes(), isize_layout);
                    self.exact_div(val, size, dest)?;
                }
            }

            "transmute" => {
                self.copy_op_transmute(args[0], dest)?;
            }
            "simd_insert" => {
                let index = u64::from(self.read_scalar(args[1])?.to_u32()?);
                let elem = args[2];
                let input = args[0];
                let (len, e_ty) = input.layout.ty.simd_size_and_type(self.tcx.tcx);
                assert!(
                    index < len,
                    "Index `{}` must be in bounds of vector type `{}`: `[0, {})`",
                    index, e_ty, len
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
                    let value = if i == index {
                        elem
                    } else {
                        self.operand_field(input, i)?
                    };
                    self.copy_op(value, place)?;
                }
            }
            "simd_extract" => {
                let index = u64::from(self.read_scalar(args[1])?.to_u32()?);
                let (len, e_ty) = args[0].layout.ty.simd_size_and_type(self.tcx.tcx);
                assert!(
                    index < len,
                    "index `{}` is out-of-bounds of vector type `{}` with length `{}`",
                    index, e_ty, len
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
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, M::PointerTag>],
        _ret: Option<(PlaceTy<'tcx, M::PointerTag>, mir::BasicBlock)>,
    ) -> InterpResult<'tcx, bool> {
        let def_id = instance.def_id();
        if Some(def_id) == self.tcx.lang_items().panic_fn() {
            // &'static str, &core::panic::Location { &'static str, u32, u32 }
            assert!(args.len() == 2);

            let msg_place = self.deref_operand(args[0])?;
            let msg = Symbol::intern(self.read_str(msg_place)?);

            let location = self.deref_operand(args[1])?;
            let (file, line, col) = (
                self.mplace_field(location, 0)?,
                self.mplace_field(location, 1)?,
                self.mplace_field(location, 2)?,
            );

            let file_place = self.deref_operand(file.into())?;
            let file = Symbol::intern(self.read_str(file_place)?);
            let line = self.read_scalar(line.into())?.to_u32()?;
            let col = self.read_scalar(col.into())?.to_u32()?;
            throw_panic!(Panic { msg, file, line, col })
        } else if Some(def_id) == self.tcx.lang_items().begin_panic_fn() {
            assert!(args.len() == 2);
            // &'static str, &(&'static str, u32, u32)
            let msg = args[0];
            let place = self.deref_operand(args[1])?;
            let (file, line, col) = (
                self.mplace_field(place, 0)?,
                self.mplace_field(place, 1)?,
                self.mplace_field(place, 2)?,
            );

            let msg_place = self.deref_operand(msg.into())?;
            let msg = Symbol::intern(self.read_str(msg_place)?);
            let file_place = self.deref_operand(file.into())?;
            let file = Symbol::intern(self.read_str(file_place)?);
            let line = self.read_scalar(line.into())?.to_u32()?;
            let col = self.read_scalar(col.into())?.to_u32()?;
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
            let b = b.to_scalar().unwrap();
            if b == minus1 {
                throw_ub_format!("exact_div: result of dividing MIN by -1 cannot be represented")
            } else {
                throw_ub_format!(
                    "exact_div: {} cannot be divided by {} without remainder",
                    a.to_scalar().unwrap(),
                    b,
                )
            }
        }
        self.binop_ignore_overflow(BinOp::Div, a, b, dest)
    }
}
