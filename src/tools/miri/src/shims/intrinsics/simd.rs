use rustc_apfloat::Float;
use rustc_middle::ty::layout::{HasParamEnv, LayoutOf};
use rustc_middle::{mir, ty, ty::FloatTy};
use rustc_target::abi::{Endian, HasDataLayout, Size};

use crate::*;
use helpers::check_arg_count;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    /// Calls the simd intrinsic `intrinsic`; the `simd_` prefix has already been removed.
    fn emulate_simd_intrinsic(
        &mut self,
        intrinsic_name: &str,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        match intrinsic_name {
            #[rustfmt::skip]
            | "neg"
            | "fabs"
            | "ceil"
            | "floor"
            | "round"
            | "trunc"
            | "fsqrt" => {
                let [op] = check_arg_count(args)?;
                let (op, op_len) = this.operand_to_simd(op)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, op_len);

                #[derive(Copy, Clone)]
                enum HostFloatOp {
                    Ceil,
                    Floor,
                    Round,
                    Trunc,
                    Sqrt,
                }
                #[derive(Copy, Clone)]
                enum Op {
                    MirOp(mir::UnOp),
                    Abs,
                    HostOp(HostFloatOp),
                }
                let which = match intrinsic_name {
                    "neg" => Op::MirOp(mir::UnOp::Neg),
                    "fabs" => Op::Abs,
                    "ceil" => Op::HostOp(HostFloatOp::Ceil),
                    "floor" => Op::HostOp(HostFloatOp::Floor),
                    "round" => Op::HostOp(HostFloatOp::Round),
                    "trunc" => Op::HostOp(HostFloatOp::Trunc),
                    "fsqrt" => Op::HostOp(HostFloatOp::Sqrt),
                    _ => unreachable!(),
                };

                for i in 0..dest_len {
                    let op = this.read_immediate(&this.mplace_index(&op, i)?.into())?;
                    let dest = this.mplace_index(&dest, i)?;
                    let val = match which {
                        Op::MirOp(mir_op) => this.unary_op(mir_op, &op)?.to_scalar(),
                        Op::Abs => {
                            // Works for f32 and f64.
                            let ty::Float(float_ty) = op.layout.ty.kind() else {
                                span_bug!(this.cur_span(), "{} operand is not a float", intrinsic_name)
                            };
                            let op = op.to_scalar();
                            match float_ty {
                                FloatTy::F32 => Scalar::from_f32(op.to_f32()?.abs()),
                                FloatTy::F64 => Scalar::from_f64(op.to_f64()?.abs()),
                            }
                        }
                        Op::HostOp(host_op) => {
                            let ty::Float(float_ty) = op.layout.ty.kind() else {
                                span_bug!(this.cur_span(), "{} operand is not a float", intrinsic_name)
                            };
                            // FIXME using host floats
                            match float_ty {
                                FloatTy::F32 => {
                                    let f = f32::from_bits(op.to_scalar().to_u32()?);
                                    let res = match host_op {
                                        HostFloatOp::Ceil => f.ceil(),
                                        HostFloatOp::Floor => f.floor(),
                                        HostFloatOp::Round => f.round(),
                                        HostFloatOp::Trunc => f.trunc(),
                                        HostFloatOp::Sqrt => f.sqrt(),
                                    };
                                    Scalar::from_u32(res.to_bits())
                                }
                                FloatTy::F64 => {
                                    let f = f64::from_bits(op.to_scalar().to_u64()?);
                                    let res = match host_op {
                                        HostFloatOp::Ceil => f.ceil(),
                                        HostFloatOp::Floor => f.floor(),
                                        HostFloatOp::Round => f.round(),
                                        HostFloatOp::Trunc => f.trunc(),
                                        HostFloatOp::Sqrt => f.sqrt(),
                                    };
                                    Scalar::from_u64(res.to_bits())
                                }
                            }

                        }
                    };
                    this.write_scalar(val, &dest.into())?;
                }
            }
            #[rustfmt::skip]
            | "add"
            | "sub"
            | "mul"
            | "div"
            | "rem"
            | "shl"
            | "shr"
            | "and"
            | "or"
            | "xor"
            | "eq"
            | "ne"
            | "lt"
            | "le"
            | "gt"
            | "ge"
            | "fmax"
            | "fmin"
            | "saturating_add"
            | "saturating_sub"
            | "arith_offset" => {
                use mir::BinOp;

                let [left, right] = check_arg_count(args)?;
                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, left_len);
                assert_eq!(dest_len, right_len);

                enum Op {
                    MirOp(BinOp),
                    SaturatingOp(BinOp),
                    FMax,
                    FMin,
                    WrappingOffset,
                }
                let which = match intrinsic_name {
                    "add" => Op::MirOp(BinOp::Add),
                    "sub" => Op::MirOp(BinOp::Sub),
                    "mul" => Op::MirOp(BinOp::Mul),
                    "div" => Op::MirOp(BinOp::Div),
                    "rem" => Op::MirOp(BinOp::Rem),
                    "shl" => Op::MirOp(BinOp::Shl),
                    "shr" => Op::MirOp(BinOp::Shr),
                    "and" => Op::MirOp(BinOp::BitAnd),
                    "or" => Op::MirOp(BinOp::BitOr),
                    "xor" => Op::MirOp(BinOp::BitXor),
                    "eq" => Op::MirOp(BinOp::Eq),
                    "ne" => Op::MirOp(BinOp::Ne),
                    "lt" => Op::MirOp(BinOp::Lt),
                    "le" => Op::MirOp(BinOp::Le),
                    "gt" => Op::MirOp(BinOp::Gt),
                    "ge" => Op::MirOp(BinOp::Ge),
                    "fmax" => Op::FMax,
                    "fmin" => Op::FMin,
                    "saturating_add" => Op::SaturatingOp(BinOp::Add),
                    "saturating_sub" => Op::SaturatingOp(BinOp::Sub),
                    "arith_offset" => Op::WrappingOffset,
                    _ => unreachable!(),
                };

                for i in 0..dest_len {
                    let left = this.read_immediate(&this.mplace_index(&left, i)?.into())?;
                    let right = this.read_immediate(&this.mplace_index(&right, i)?.into())?;
                    let dest = this.mplace_index(&dest, i)?;
                    let val = match which {
                        Op::MirOp(mir_op) => {
                            let (val, overflowed, ty) = this.overflowing_binary_op(mir_op, &left, &right)?;
                            if matches!(mir_op, BinOp::Shl | BinOp::Shr) {
                                // Shifts have extra UB as SIMD operations that the MIR binop does not have.
                                // See <https://github.com/rust-lang/rust/issues/91237>.
                                if overflowed {
                                    let r_val = right.to_scalar().to_bits(right.layout.size)?;
                                    throw_ub_format!("overflowing shift by {r_val} in `simd_{intrinsic_name}` in SIMD lane {i}");
                                }
                            }
                            if matches!(mir_op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
                                // Special handling for boolean-returning operations
                                assert_eq!(ty, this.tcx.types.bool);
                                let val = val.to_bool().unwrap();
                                bool_to_simd_element(val, dest.layout.size)
                            } else {
                                assert_ne!(ty, this.tcx.types.bool);
                                assert_eq!(ty, dest.layout.ty);
                                val
                            }
                        }
                        Op::SaturatingOp(mir_op) => {
                            this.saturating_arith(mir_op, &left, &right)?
                        }
                        Op::WrappingOffset => {
                            let ptr = left.to_scalar().to_pointer(this)?;
                            let offset_count = right.to_scalar().to_machine_isize(this)?;
                            let pointee_ty = left.layout.ty.builtin_deref(true).unwrap().ty;

                            let pointee_size = i64::try_from(this.layout_of(pointee_ty)?.size.bytes()).unwrap();
                            let offset_bytes = offset_count.wrapping_mul(pointee_size);
                            let offset_ptr = ptr.wrapping_signed_offset(offset_bytes, this);
                            Scalar::from_maybe_pointer(offset_ptr, this)
                        }
                        Op::FMax => {
                            fmax_op(&left, &right)?
                        }
                        Op::FMin => {
                            fmin_op(&left, &right)?
                        }
                    };
                    this.write_scalar(val, &dest.into())?;
                }
            }
            "fma" => {
                let [a, b, c] = check_arg_count(args)?;
                let (a, a_len) = this.operand_to_simd(a)?;
                let (b, b_len) = this.operand_to_simd(b)?;
                let (c, c_len) = this.operand_to_simd(c)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, a_len);
                assert_eq!(dest_len, b_len);
                assert_eq!(dest_len, c_len);

                for i in 0..dest_len {
                    let a = this.read_scalar(&this.mplace_index(&a, i)?.into())?;
                    let b = this.read_scalar(&this.mplace_index(&b, i)?.into())?;
                    let c = this.read_scalar(&this.mplace_index(&c, i)?.into())?;
                    let dest = this.mplace_index(&dest, i)?;

                    // Works for f32 and f64.
                    // FIXME: using host floats to work around https://github.com/rust-lang/miri/issues/2468.
                    let ty::Float(float_ty) = dest.layout.ty.kind() else {
                        span_bug!(this.cur_span(), "{} operand is not a float", intrinsic_name)
                    };
                    let val = match float_ty {
                        FloatTy::F32 => {
                            let a = f32::from_bits(a.to_u32()?);
                            let b = f32::from_bits(b.to_u32()?);
                            let c = f32::from_bits(c.to_u32()?);
                            let res = a.mul_add(b, c);
                            Scalar::from_u32(res.to_bits())
                        }
                        FloatTy::F64 => {
                            let a = f64::from_bits(a.to_u64()?);
                            let b = f64::from_bits(b.to_u64()?);
                            let c = f64::from_bits(c.to_u64()?);
                            let res = a.mul_add(b, c);
                            Scalar::from_u64(res.to_bits())
                        }
                    };
                    this.write_scalar(val, &dest.into())?;
                }
            }
            #[rustfmt::skip]
            | "reduce_and"
            | "reduce_or"
            | "reduce_xor"
            | "reduce_any"
            | "reduce_all"
            | "reduce_max"
            | "reduce_min" => {
                use mir::BinOp;

                let [op] = check_arg_count(args)?;
                let (op, op_len) = this.operand_to_simd(op)?;

                let imm_from_bool =
                    |b| ImmTy::from_scalar(Scalar::from_bool(b), this.machine.layouts.bool);

                enum Op {
                    MirOp(BinOp),
                    MirOpBool(BinOp),
                    Max,
                    Min,
                }
                let which = match intrinsic_name {
                    "reduce_and" => Op::MirOp(BinOp::BitAnd),
                    "reduce_or" => Op::MirOp(BinOp::BitOr),
                    "reduce_xor" => Op::MirOp(BinOp::BitXor),
                    "reduce_any" => Op::MirOpBool(BinOp::BitOr),
                    "reduce_all" => Op::MirOpBool(BinOp::BitAnd),
                    "reduce_max" => Op::Max,
                    "reduce_min" => Op::Min,
                    _ => unreachable!(),
                };

                // Initialize with first lane, then proceed with the rest.
                let mut res = this.read_immediate(&this.mplace_index(&op, 0)?.into())?;
                if matches!(which, Op::MirOpBool(_)) {
                    // Convert to `bool` scalar.
                    res = imm_from_bool(simd_element_to_bool(res)?);
                }
                for i in 1..op_len {
                    let op = this.read_immediate(&this.mplace_index(&op, i)?.into())?;
                    res = match which {
                        Op::MirOp(mir_op) => {
                            this.binary_op(mir_op, &res, &op)?
                        }
                        Op::MirOpBool(mir_op) => {
                            let op = imm_from_bool(simd_element_to_bool(op)?);
                            this.binary_op(mir_op, &res, &op)?
                        }
                        Op::Max => {
                            if matches!(res.layout.ty.kind(), ty::Float(_)) {
                                ImmTy::from_scalar(fmax_op(&res, &op)?, res.layout)
                            } else {
                                // Just boring integers, so NaNs to worry about
                                if this.binary_op(BinOp::Ge, &res, &op)?.to_scalar().to_bool()? {
                                    res
                                } else {
                                    op
                                }
                            }
                        }
                        Op::Min => {
                            if matches!(res.layout.ty.kind(), ty::Float(_)) {
                                ImmTy::from_scalar(fmin_op(&res, &op)?, res.layout)
                            } else {
                                // Just boring integers, so NaNs to worry about
                                if this.binary_op(BinOp::Le, &res, &op)?.to_scalar().to_bool()? {
                                    res
                                } else {
                                    op
                                }
                            }
                        }
                    };
                }
                this.write_immediate(*res, dest)?;
            }
            #[rustfmt::skip]
            | "reduce_add_ordered"
            | "reduce_mul_ordered" => {
                use mir::BinOp;

                let [op, init] = check_arg_count(args)?;
                let (op, op_len) = this.operand_to_simd(op)?;
                let init = this.read_immediate(init)?;

                let mir_op = match intrinsic_name {
                    "reduce_add_ordered" => BinOp::Add,
                    "reduce_mul_ordered" => BinOp::Mul,
                    _ => unreachable!(),
                };

                let mut res = init;
                for i in 0..op_len {
                    let op = this.read_immediate(&this.mplace_index(&op, i)?.into())?;
                    res = this.binary_op(mir_op, &res, &op)?;
                }
                this.write_immediate(*res, dest)?;
            }
            "select" => {
                let [mask, yes, no] = check_arg_count(args)?;
                let (mask, mask_len) = this.operand_to_simd(mask)?;
                let (yes, yes_len) = this.operand_to_simd(yes)?;
                let (no, no_len) = this.operand_to_simd(no)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, mask_len);
                assert_eq!(dest_len, yes_len);
                assert_eq!(dest_len, no_len);

                for i in 0..dest_len {
                    let mask = this.read_immediate(&this.mplace_index(&mask, i)?.into())?;
                    let yes = this.read_immediate(&this.mplace_index(&yes, i)?.into())?;
                    let no = this.read_immediate(&this.mplace_index(&no, i)?.into())?;
                    let dest = this.mplace_index(&dest, i)?;

                    let val = if simd_element_to_bool(mask)? { yes } else { no };
                    this.write_immediate(*val, &dest.into())?;
                }
            }
            "select_bitmask" => {
                let [mask, yes, no] = check_arg_count(args)?;
                let (yes, yes_len) = this.operand_to_simd(yes)?;
                let (no, no_len) = this.operand_to_simd(no)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;
                let bitmask_len = dest_len.max(8);

                assert!(mask.layout.ty.is_integral());
                assert!(bitmask_len <= 64);
                assert_eq!(bitmask_len, mask.layout.size.bits());
                assert_eq!(dest_len, yes_len);
                assert_eq!(dest_len, no_len);
                let dest_len = u32::try_from(dest_len).unwrap();
                let bitmask_len = u32::try_from(bitmask_len).unwrap();

                let mask: u64 =
                    this.read_scalar(mask)?.to_bits(mask.layout.size)?.try_into().unwrap();
                for i in 0..dest_len {
                    let mask = mask
                        & 1u64
                            .checked_shl(simd_bitmask_index(i, dest_len, this.data_layout().endian))
                            .unwrap();
                    let yes = this.read_immediate(&this.mplace_index(&yes, i.into())?.into())?;
                    let no = this.read_immediate(&this.mplace_index(&no, i.into())?.into())?;
                    let dest = this.mplace_index(&dest, i.into())?;

                    let val = if mask != 0 { yes } else { no };
                    this.write_immediate(*val, &dest.into())?;
                }
                for i in dest_len..bitmask_len {
                    // If the mask is "padded", ensure that padding is all-zero.
                    let mask = mask & 1u64.checked_shl(i).unwrap();
                    if mask != 0 {
                        throw_ub_format!(
                            "a SIMD bitmask less than 8 bits long must be filled with 0s for the remaining bits"
                        );
                    }
                }
            }
            #[rustfmt::skip]
            "cast" | "as" => {
                let [op] = check_arg_count(args)?;
                let (op, op_len) = this.operand_to_simd(op)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, op_len);

                let safe_cast = intrinsic_name == "as";

                for i in 0..dest_len {
                    let op = this.read_immediate(&this.mplace_index(&op, i)?.into())?;
                    let dest = this.mplace_index(&dest, i)?;

                    let val = match (op.layout.ty.kind(), dest.layout.ty.kind()) {
                        // Int-to-(int|float): always safe
                        (ty::Int(_) | ty::Uint(_), ty::Int(_) | ty::Uint(_) | ty::Float(_)) =>
                            this.int_to_int_or_float(&op, dest.layout.ty)?,
                        // Float-to-float: always safe
                        (ty::Float(_), ty::Float(_)) =>
                            this.float_to_float_or_int(&op, dest.layout.ty)?,
                        // Float-to-int in safe mode
                        (ty::Float(_), ty::Int(_) | ty::Uint(_)) if safe_cast =>
                            this.float_to_float_or_int(&op, dest.layout.ty)?,
                        // Float-to-int in unchecked mode
                        (ty::Float(FloatTy::F32), ty::Int(_) | ty::Uint(_)) if !safe_cast =>
                            this.float_to_int_unchecked(op.to_scalar().to_f32()?, dest.layout.ty)?.into(),
                        (ty::Float(FloatTy::F64), ty::Int(_) | ty::Uint(_)) if !safe_cast =>
                            this.float_to_int_unchecked(op.to_scalar().to_f64()?, dest.layout.ty)?.into(),
                        _ =>
                            throw_unsup_format!(
                                "Unsupported SIMD cast from element type {from_ty} to {to_ty}",
                                from_ty = op.layout.ty,
                                to_ty = dest.layout.ty,
                            ),
                    };
                    this.write_immediate(val, &dest.into())?;
                }
            }
            "shuffle" => {
                let [left, right, index] = check_arg_count(args)?;
                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                // `index` is an array, not a SIMD type
                let ty::Array(_, index_len) = index.layout.ty.kind() else {
                    span_bug!(this.cur_span(), "simd_shuffle index argument has non-array type {}", index.layout.ty)
                };
                let index_len = index_len.eval_target_usize(*this.tcx, this.param_env());

                assert_eq!(left_len, right_len);
                assert_eq!(index_len, dest_len);

                for i in 0..dest_len {
                    let src_index: u64 = this
                        .read_immediate(&this.operand_index(index, i)?)?
                        .to_scalar()
                        .to_u32()?
                        .into();
                    let dest = this.mplace_index(&dest, i)?;

                    let val = if src_index < left_len {
                        this.read_immediate(&this.mplace_index(&left, src_index)?.into())?
                    } else if src_index < left_len.checked_add(right_len).unwrap() {
                        let right_idx = src_index.checked_sub(left_len).unwrap();
                        this.read_immediate(&this.mplace_index(&right, right_idx)?.into())?
                    } else {
                        span_bug!(
                            this.cur_span(),
                            "simd_shuffle index {src_index} is out of bounds for 2 vectors of size {left_len}",
                        );
                    };
                    this.write_immediate(*val, &dest.into())?;
                }
            }
            "gather" => {
                let [passthru, ptrs, mask] = check_arg_count(args)?;
                let (passthru, passthru_len) = this.operand_to_simd(passthru)?;
                let (ptrs, ptrs_len) = this.operand_to_simd(ptrs)?;
                let (mask, mask_len) = this.operand_to_simd(mask)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, passthru_len);
                assert_eq!(dest_len, ptrs_len);
                assert_eq!(dest_len, mask_len);

                for i in 0..dest_len {
                    let passthru = this.read_immediate(&this.mplace_index(&passthru, i)?.into())?;
                    let ptr = this.read_immediate(&this.mplace_index(&ptrs, i)?.into())?;
                    let mask = this.read_immediate(&this.mplace_index(&mask, i)?.into())?;
                    let dest = this.mplace_index(&dest, i)?;

                    let val = if simd_element_to_bool(mask)? {
                        let place = this.deref_operand(&ptr.into())?;
                        this.read_immediate(&place.into())?
                    } else {
                        passthru
                    };
                    this.write_immediate(*val, &dest.into())?;
                }
            }
            "scatter" => {
                let [value, ptrs, mask] = check_arg_count(args)?;
                let (value, value_len) = this.operand_to_simd(value)?;
                let (ptrs, ptrs_len) = this.operand_to_simd(ptrs)?;
                let (mask, mask_len) = this.operand_to_simd(mask)?;

                assert_eq!(ptrs_len, value_len);
                assert_eq!(ptrs_len, mask_len);

                for i in 0..ptrs_len {
                    let value = this.read_immediate(&this.mplace_index(&value, i)?.into())?;
                    let ptr = this.read_immediate(&this.mplace_index(&ptrs, i)?.into())?;
                    let mask = this.read_immediate(&this.mplace_index(&mask, i)?.into())?;

                    if simd_element_to_bool(mask)? {
                        let place = this.deref_operand(&ptr.into())?;
                        this.write_immediate(*value, &place.into())?;
                    }
                }
            }
            "bitmask" => {
                let [op] = check_arg_count(args)?;
                let (op, op_len) = this.operand_to_simd(op)?;
                let bitmask_len = op_len.max(8);

                assert!(dest.layout.ty.is_integral());
                assert!(bitmask_len <= 64);
                assert_eq!(bitmask_len, dest.layout.size.bits());
                let op_len = u32::try_from(op_len).unwrap();

                let mut res = 0u64;
                for i in 0..op_len {
                    let op = this.read_immediate(&this.mplace_index(&op, i.into())?.into())?;
                    if simd_element_to_bool(op)? {
                        res |= 1u64
                            .checked_shl(simd_bitmask_index(i, op_len, this.data_layout().endian))
                            .unwrap();
                    }
                }
                this.write_int(res, dest)?;
            }

            name => throw_unsup_format!("unimplemented intrinsic: `simd_{name}`"),
        }
        Ok(())
    }
}

fn bool_to_simd_element(b: bool, size: Size) -> Scalar<Provenance> {
    // SIMD uses all-1 as pattern for "true"
    let val = if b { -1 } else { 0 };
    Scalar::from_int(val, size)
}

fn simd_element_to_bool(elem: ImmTy<'_, Provenance>) -> InterpResult<'_, bool> {
    let val = elem.to_scalar().to_int(elem.layout.size)?;
    Ok(match val {
        0 => false,
        -1 => true,
        _ => throw_ub_format!("each element of a SIMD mask must be all-0-bits or all-1-bits"),
    })
}

fn simd_bitmask_index(idx: u32, vec_len: u32, endianess: Endian) -> u32 {
    assert!(idx < vec_len);
    match endianess {
        Endian::Little => idx,
        #[allow(clippy::integer_arithmetic)] // idx < vec_len
        Endian::Big => vec_len - 1 - idx, // reverse order of bits
    }
}

fn fmax_op<'tcx>(
    left: &ImmTy<'tcx, Provenance>,
    right: &ImmTy<'tcx, Provenance>,
) -> InterpResult<'tcx, Scalar<Provenance>> {
    assert_eq!(left.layout.ty, right.layout.ty);
    let ty::Float(float_ty) = left.layout.ty.kind() else {
        bug!("fmax operand is not a float")
    };
    let left = left.to_scalar();
    let right = right.to_scalar();
    Ok(match float_ty {
        FloatTy::F32 => Scalar::from_f32(left.to_f32()?.max(right.to_f32()?)),
        FloatTy::F64 => Scalar::from_f64(left.to_f64()?.max(right.to_f64()?)),
    })
}

fn fmin_op<'tcx>(
    left: &ImmTy<'tcx, Provenance>,
    right: &ImmTy<'tcx, Provenance>,
) -> InterpResult<'tcx, Scalar<Provenance>> {
    assert_eq!(left.layout.ty, right.layout.ty);
    let ty::Float(float_ty) = left.layout.ty.kind() else {
        bug!("fmin operand is not a float")
    };
    let left = left.to_scalar();
    let right = right.to_scalar();
    Ok(match float_ty {
        FloatTy::F32 => Scalar::from_f32(left.to_f32()?.min(right.to_f32()?)),
        FloatTy::F64 => Scalar::from_f64(left.to_f64()?.min(right.to_f64()?)),
    })
}
