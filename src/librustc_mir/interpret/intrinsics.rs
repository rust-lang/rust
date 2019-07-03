//! Intrinsics and other functions that the miri engine executes without
//! looking at their MIR. Intrinsics/functions supported here are shared by CTFE
//! and miri.

use syntax::symbol::Symbol;
use rustc::ty;
use rustc::ty::layout::{LayoutOf, Primitive, Size};
use rustc::mir::BinOp;
use rustc::mir::interpret::{
    InterpResult, InterpError, Scalar,
};

use super::{
    Machine, PlaceTy, OpTy, InterpCx, Immediate,
};

mod type_name;

pub use type_name::*;

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

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Returns `true` if emulation happened.
    pub fn emulate_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, M::PointerTag>],
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, bool> {
        let substs = instance.substs;

        let intrinsic_name = &self.tcx.item_name(instance.def_id()).as_str()[..];
        match intrinsic_name {
            "min_align_of" => {
                let elem_ty = substs.type_at(0);
                let elem_align = self.layout_of(elem_ty)?.align.abi.bytes();
                let align_val = Scalar::from_uint(elem_align, dest.layout.size);
                self.write_scalar(align_val, dest)?;
            }

            "needs_drop" => {
                let ty = substs.type_at(0);
                let ty_needs_drop = ty.needs_drop(self.tcx.tcx, self.param_env);
                let val = Scalar::from_bool(ty_needs_drop);
                self.write_scalar(val, dest)?;
            }

            "size_of" => {
                let ty = substs.type_at(0);
                let size = self.layout_of(ty)?.size.bytes() as u128;
                let size_val = Scalar::from_uint(size, dest.layout.size);
                self.write_scalar(size_val, dest)?;
            }

            "type_id" => {
                let ty = substs.type_at(0);
                let type_id = self.tcx.type_id_hash(ty) as u128;
                let id_val = Scalar::from_uint(type_id, dest.layout.size);
                self.write_scalar(id_val, dest)?;
            }

            "type_name" => {
                let alloc = alloc_type_name(self.tcx.tcx, substs.type_at(0));
                let name_id = self.tcx.alloc_map.lock().create_memory_alloc(alloc);
                let id_ptr = self.memory.tag_static_base_pointer(name_id.into());
                let alloc_len = alloc.bytes.len() as u64;
                let name_val = Immediate::new_slice(Scalar::Ptr(id_ptr), alloc_len, self);
                self.write_immediate(name_val, dest)?;
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
                let bits = self.read_scalar(args[0])?.to_bits(layout_of.size)?;
                let kind = match layout_of.abi {
                    ty::layout::Abi::Scalar(ref scalar) => scalar.value,
                    _ => Err(::rustc::mir::interpret::InterpError::TypeNotPrimitive(ty))?,
                };
                let out_val = if intrinsic_name.ends_with("_nonzero") {
                    if bits == 0 {
                        return err!(Intrinsic(format!("{} called on 0", intrinsic_name)));
                    }
                    numeric_intrinsic(intrinsic_name.trim_end_matches("_nonzero"), bits, kind)?
                } else {
                    numeric_intrinsic(intrinsic_name, bits, kind)?
                };
                self.write_scalar(out_val, dest)?;
            }
            | "overflowing_add"
            | "overflowing_sub"
            | "overflowing_mul"
            | "add_with_overflow"
            | "sub_with_overflow"
            | "mul_with_overflow" => {
                let lhs = self.read_immediate(args[0])?;
                let rhs = self.read_immediate(args[1])?;
                let (bin_op, ignore_overflow) = match intrinsic_name {
                    "overflowing_add" => (BinOp::Add, true),
                    "overflowing_sub" => (BinOp::Sub, true),
                    "overflowing_mul" => (BinOp::Mul, true),
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
                let (val, overflowed) = self.binary_op(if is_add {
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
                        let first_term: u128 = l.to_scalar()?.to_bits(l.layout.size)?;
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
                let (val, overflowed) = self.binary_op(bin_op, l, r)?;
                if overflowed {
                    let layout = self.layout_of(substs.type_at(0))?;
                    let r_val =  r.to_scalar()?.to_bits(layout.size)?;
                    return err!(Intrinsic(
                        format!("Overflowing shift by {} in {}", r_val, intrinsic_name),
                    ));
                }
                self.write_scalar(val, dest)?;
            }
            "rotate_left" | "rotate_right" => {
                // rotate_left: (X << (S % BW)) | (X >> ((BW - S) % BW))
                // rotate_right: (X << ((BW - S) % BW)) | (X >> (S % BW))
                let layout = self.layout_of(substs.type_at(0))?;
                let val_bits = self.read_scalar(args[0])?.to_bits(layout.size)?;
                let raw_shift_bits = self.read_scalar(args[1])?.to_bits(layout.size)?;
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
            "transmute" => {
                self.copy_op_transmute(args[0], dest)?;
            }

            _ => return Ok(false),
        }

        Ok(true)
    }

    /// "Intercept" a function call because we have something special to do for it.
    /// Returns `true` if an intercept happened.
    pub fn hook_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, M::PointerTag>],
        dest: Option<PlaceTy<'tcx, M::PointerTag>>,
    ) -> InterpResult<'tcx, bool> {
        let def_id = instance.def_id();
        // Some fn calls are actually BinOp intrinsics
        if let Some((op, oflo)) = self.tcx.is_binop_lang_item(def_id) {
            let dest = dest.expect("128 lowerings can't diverge");
            let l = self.read_immediate(args[0])?;
            let r = self.read_immediate(args[1])?;
            if oflo {
                self.binop_with_overflow(op, l, r, dest)?;
            } else {
                self.binop_ignore_overflow(op, l, r, dest)?;
            }
            return Ok(true);
        } else if Some(def_id) == self.tcx.lang_items().panic_fn() {
            assert!(args.len() == 1);
            // &(&'static str, &'static str, u32, u32)
            let place = self.deref_operand(args[0])?;
            let (msg, file, line, col) = (
                self.mplace_field(place, 0)?,
                self.mplace_field(place, 1)?,
                self.mplace_field(place, 2)?,
                self.mplace_field(place, 3)?,
            );

            let msg_place = self.deref_operand(msg.into())?;
            let msg = Symbol::intern(self.read_str(msg_place)?);
            let file_place = self.deref_operand(file.into())?;
            let file = Symbol::intern(self.read_str(file_place)?);
            let line = self.read_scalar(line.into())?.to_u32()?;
            let col = self.read_scalar(col.into())?.to_u32()?;
            return Err(InterpError::Panic { msg, file, line, col }.into());
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
            return Err(InterpError::Panic { msg, file, line, col }.into());
        } else {
            return Ok(false);
        }
    }
}
