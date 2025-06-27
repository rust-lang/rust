use either::Either;
use rand::Rng;
use rustc_abi::{Endian, HasDataLayout};
use rustc_apfloat::{Float, Round};
use rustc_middle::ty::FloatTy;
use rustc_middle::{mir, ty};
use rustc_span::{Symbol, sym};

use crate::helpers::{
    ToHost, ToSoft, bool_to_simd_element, check_intrinsic_arg_count, simd_element_to_bool,
};
use crate::*;

#[derive(Copy, Clone)]
pub(crate) enum MinMax {
    Min,
    Max,
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Calls the simd intrinsic `intrinsic`; the `simd_` prefix has already been removed.
    /// Returns `Ok(true)` if the intrinsic was handled.
    fn emulate_simd_intrinsic(
        &mut self,
        intrinsic_name: &str,
        generic_args: ty::GenericArgsRef<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        match intrinsic_name {
            #[rustfmt::skip]
            | "neg"
            | "fabs"
            | "ceil"
            | "floor"
            | "round"
            | "trunc"
            | "fsqrt"
            | "fsin"
            | "fcos"
            | "fexp"
            | "fexp2"
            | "flog"
            | "flog2"
            | "flog10"
            | "ctlz"
            | "ctpop"
            | "cttz"
            | "bswap"
            | "bitreverse"
            => {
                let [op] = check_intrinsic_arg_count(args)?;
                let (op, op_len) = this.project_to_simd(op)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, op_len);

                #[derive(Copy, Clone)]
                enum Op<'a> {
                    MirOp(mir::UnOp),
                    Abs,
                    Round(rustc_apfloat::Round),
                    Numeric(Symbol),
                    HostOp(&'a str),
                }
                let which = match intrinsic_name {
                    "neg" => Op::MirOp(mir::UnOp::Neg),
                    "fabs" => Op::Abs,
                    "ceil" => Op::Round(rustc_apfloat::Round::TowardPositive),
                    "floor" => Op::Round(rustc_apfloat::Round::TowardNegative),
                    "round" => Op::Round(rustc_apfloat::Round::NearestTiesToAway),
                    "trunc" => Op::Round(rustc_apfloat::Round::TowardZero),
                    "ctlz" => Op::Numeric(sym::ctlz),
                    "ctpop" => Op::Numeric(sym::ctpop),
                    "cttz" => Op::Numeric(sym::cttz),
                    "bswap" => Op::Numeric(sym::bswap),
                    "bitreverse" => Op::Numeric(sym::bitreverse),
                    _ => Op::HostOp(intrinsic_name),
                };

                for i in 0..dest_len {
                    let op = this.read_immediate(&this.project_index(&op, i)?)?;
                    let dest = this.project_index(&dest, i)?;
                    let val = match which {
                        Op::MirOp(mir_op) => {
                            // This already does NaN adjustments
                            this.unary_op(mir_op, &op)?.to_scalar()
                        }
                        Op::Abs => {
                            // Works for f32 and f64.
                            let ty::Float(float_ty) = op.layout.ty.kind() else {
                                span_bug!(this.cur_span(), "{} operand is not a float", intrinsic_name)
                            };
                            let op = op.to_scalar();
                            // "Bitwise" operation, no NaN adjustments
                            match float_ty {
                                FloatTy::F16 => unimplemented!("f16_f128"),
                                FloatTy::F32 => Scalar::from_f32(op.to_f32()?.abs()),
                                FloatTy::F64 => Scalar::from_f64(op.to_f64()?.abs()),
                                FloatTy::F128 => unimplemented!("f16_f128"),
                            }
                        }
                        Op::HostOp(host_op) => {
                            let ty::Float(float_ty) = op.layout.ty.kind() else {
                                span_bug!(this.cur_span(), "{} operand is not a float", intrinsic_name)
                            };
                            // Using host floats except for sqrt (but it's fine, these operations do not
                            // have guaranteed precision).
                            match float_ty {
                                FloatTy::F16 => unimplemented!("f16_f128"),
                                FloatTy::F32 => {
                                    let f = op.to_scalar().to_f32()?;
                                    let res = match host_op {
                                        "fsqrt" => math::sqrt(f),
                                        "fsin" => f.to_host().sin().to_soft(),
                                        "fcos" => f.to_host().cos().to_soft(),
                                        "fexp" => f.to_host().exp().to_soft(),
                                        "fexp2" => f.to_host().exp2().to_soft(),
                                        "flog" => f.to_host().ln().to_soft(),
                                        "flog2" => f.to_host().log2().to_soft(),
                                        "flog10" => f.to_host().log10().to_soft(),
                                        _ => bug!(),
                                    };
                                    let res = this.adjust_nan(res, &[f]);
                                    Scalar::from(res)
                                }
                                FloatTy::F64 => {
                                    let f = op.to_scalar().to_f64()?;
                                    let res = match host_op {
                                        "fsqrt" => math::sqrt(f),
                                        "fsin" => f.to_host().sin().to_soft(),
                                        "fcos" => f.to_host().cos().to_soft(),
                                        "fexp" => f.to_host().exp().to_soft(),
                                        "fexp2" => f.to_host().exp2().to_soft(),
                                        "flog" => f.to_host().ln().to_soft(),
                                        "flog2" => f.to_host().log2().to_soft(),
                                        "flog10" => f.to_host().log10().to_soft(),
                                        _ => bug!(),
                                    };
                                    let res = this.adjust_nan(res, &[f]);
                                    Scalar::from(res)
                                }
                                FloatTy::F128 => unimplemented!("f16_f128"),
                            }
                        }
                        Op::Round(rounding) => {
                            let ty::Float(float_ty) = op.layout.ty.kind() else {
                                span_bug!(this.cur_span(), "{} operand is not a float", intrinsic_name)
                            };
                            match float_ty {
                                FloatTy::F16 => unimplemented!("f16_f128"),
                                FloatTy::F32 => {
                                    let f = op.to_scalar().to_f32()?;
                                    let res = f.round_to_integral(rounding).value;
                                    let res = this.adjust_nan(res, &[f]);
                                    Scalar::from_f32(res)
                                }
                                FloatTy::F64 => {
                                    let f = op.to_scalar().to_f64()?;
                                    let res = f.round_to_integral(rounding).value;
                                    let res = this.adjust_nan(res, &[f]);
                                    Scalar::from_f64(res)
                                }
                                FloatTy::F128 => unimplemented!("f16_f128"),
                            }
                        }
                        Op::Numeric(name) => {
                            this.numeric_intrinsic(name, op.to_scalar(), op.layout, op.layout)?
                        }
                    };
                    this.write_scalar(val, &dest)?;
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
            | "arith_offset"
            => {
                use mir::BinOp;

                let [left, right] = check_intrinsic_arg_count(args)?;
                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, left_len);
                assert_eq!(dest_len, right_len);

                enum Op {
                    MirOp(BinOp),
                    SaturatingOp(BinOp),
                    FMinMax(MinMax),
                    WrappingOffset,
                }
                let which = match intrinsic_name {
                    "add" => Op::MirOp(BinOp::Add),
                    "sub" => Op::MirOp(BinOp::Sub),
                    "mul" => Op::MirOp(BinOp::Mul),
                    "div" => Op::MirOp(BinOp::Div),
                    "rem" => Op::MirOp(BinOp::Rem),
                    "shl" => Op::MirOp(BinOp::ShlUnchecked),
                    "shr" => Op::MirOp(BinOp::ShrUnchecked),
                    "and" => Op::MirOp(BinOp::BitAnd),
                    "or" => Op::MirOp(BinOp::BitOr),
                    "xor" => Op::MirOp(BinOp::BitXor),
                    "eq" => Op::MirOp(BinOp::Eq),
                    "ne" => Op::MirOp(BinOp::Ne),
                    "lt" => Op::MirOp(BinOp::Lt),
                    "le" => Op::MirOp(BinOp::Le),
                    "gt" => Op::MirOp(BinOp::Gt),
                    "ge" => Op::MirOp(BinOp::Ge),
                    "fmax" => Op::FMinMax(MinMax::Max),
                    "fmin" => Op::FMinMax(MinMax::Min),
                    "saturating_add" => Op::SaturatingOp(BinOp::Add),
                    "saturating_sub" => Op::SaturatingOp(BinOp::Sub),
                    "arith_offset" => Op::WrappingOffset,
                    _ => unreachable!(),
                };

                for i in 0..dest_len {
                    let left = this.read_immediate(&this.project_index(&left, i)?)?;
                    let right = this.read_immediate(&this.project_index(&right, i)?)?;
                    let dest = this.project_index(&dest, i)?;
                    let val = match which {
                        Op::MirOp(mir_op) => {
                            // This does NaN adjustments.
                            let val = this.binary_op(mir_op, &left, &right).map_err_kind(|kind| {
                                match kind {
                                    InterpErrorKind::UndefinedBehavior(UndefinedBehaviorInfo::ShiftOverflow { shift_amount, .. }) => {
                                        // This resets the interpreter backtrace, but it's not worth avoiding that.
                                        let shift_amount = match shift_amount {
                                            Either::Left(v) => v.to_string(),
                                            Either::Right(v) => v.to_string(),
                                        };
                                        err_ub_format!("overflowing shift by {shift_amount} in `simd_{intrinsic_name}` in lane {i}")
                                    }
                                    kind => kind
                                }
                            })?;
                            if matches!(mir_op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
                                // Special handling for boolean-returning operations
                                assert_eq!(val.layout.ty, this.tcx.types.bool);
                                let val = val.to_scalar().to_bool().unwrap();
                                bool_to_simd_element(val, dest.layout.size)
                            } else {
                                assert_ne!(val.layout.ty, this.tcx.types.bool);
                                assert_eq!(val.layout.ty, dest.layout.ty);
                                val.to_scalar()
                            }
                        }
                        Op::SaturatingOp(mir_op) => {
                            this.saturating_arith(mir_op, &left, &right)?
                        }
                        Op::WrappingOffset => {
                            let ptr = left.to_scalar().to_pointer(this)?;
                            let offset_count = right.to_scalar().to_target_isize(this)?;
                            let pointee_ty = left.layout.ty.builtin_deref(true).unwrap();

                            let pointee_size = i64::try_from(this.layout_of(pointee_ty)?.size.bytes()).unwrap();
                            let offset_bytes = offset_count.wrapping_mul(pointee_size);
                            let offset_ptr = ptr.wrapping_signed_offset(offset_bytes, this);
                            Scalar::from_maybe_pointer(offset_ptr, this)
                        }
                        Op::FMinMax(op) => {
                            this.fminmax_op(op, &left, &right)?
                        }
                    };
                    this.write_scalar(val, &dest)?;
                }
            }
            "fma" | "relaxed_fma" => {
                let [a, b, c] = check_intrinsic_arg_count(args)?;
                let (a, a_len) = this.project_to_simd(a)?;
                let (b, b_len) = this.project_to_simd(b)?;
                let (c, c_len) = this.project_to_simd(c)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, a_len);
                assert_eq!(dest_len, b_len);
                assert_eq!(dest_len, c_len);

                for i in 0..dest_len {
                    let a = this.read_scalar(&this.project_index(&a, i)?)?;
                    let b = this.read_scalar(&this.project_index(&b, i)?)?;
                    let c = this.read_scalar(&this.project_index(&c, i)?)?;
                    let dest = this.project_index(&dest, i)?;

                    let fuse: bool = intrinsic_name == "fma"
                        || (this.machine.float_nondet && this.machine.rng.get_mut().random());

                    // Works for f32 and f64.
                    // FIXME: using host floats to work around https://github.com/rust-lang/miri/issues/2468.
                    let ty::Float(float_ty) = dest.layout.ty.kind() else {
                        span_bug!(this.cur_span(), "{} operand is not a float", intrinsic_name)
                    };
                    let val = match float_ty {
                        FloatTy::F16 => unimplemented!("f16_f128"),
                        FloatTy::F32 => {
                            let a = a.to_f32()?;
                            let b = b.to_f32()?;
                            let c = c.to_f32()?;
                            let res = if fuse {
                                a.mul_add(b, c).value
                            } else {
                                ((a * b).value + c).value
                            };
                            let res = this.adjust_nan(res, &[a, b, c]);
                            Scalar::from(res)
                        }
                        FloatTy::F64 => {
                            let a = a.to_f64()?;
                            let b = b.to_f64()?;
                            let c = c.to_f64()?;
                            let res = if fuse {
                                a.mul_add(b, c).value
                            } else {
                                ((a * b).value + c).value
                            };
                            let res = this.adjust_nan(res, &[a, b, c]);
                            Scalar::from(res)
                        }
                        FloatTy::F128 => unimplemented!("f16_f128"),
                    };
                    this.write_scalar(val, &dest)?;
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

                let [op] = check_intrinsic_arg_count(args)?;
                let (op, op_len) = this.project_to_simd(op)?;

                let imm_from_bool =
                    |b| ImmTy::from_scalar(Scalar::from_bool(b), this.machine.layouts.bool);

                enum Op {
                    MirOp(BinOp),
                    MirOpBool(BinOp),
                    MinMax(MinMax),
                }
                let which = match intrinsic_name {
                    "reduce_and" => Op::MirOp(BinOp::BitAnd),
                    "reduce_or" => Op::MirOp(BinOp::BitOr),
                    "reduce_xor" => Op::MirOp(BinOp::BitXor),
                    "reduce_any" => Op::MirOpBool(BinOp::BitOr),
                    "reduce_all" => Op::MirOpBool(BinOp::BitAnd),
                    "reduce_max" => Op::MinMax(MinMax::Max),
                    "reduce_min" => Op::MinMax(MinMax::Min),
                    _ => unreachable!(),
                };

                // Initialize with first lane, then proceed with the rest.
                let mut res = this.read_immediate(&this.project_index(&op, 0)?)?;
                if matches!(which, Op::MirOpBool(_)) {
                    // Convert to `bool` scalar.
                    res = imm_from_bool(simd_element_to_bool(res)?);
                }
                for i in 1..op_len {
                    let op = this.read_immediate(&this.project_index(&op, i)?)?;
                    res = match which {
                        Op::MirOp(mir_op) => {
                            this.binary_op(mir_op, &res, &op)?
                        }
                        Op::MirOpBool(mir_op) => {
                            let op = imm_from_bool(simd_element_to_bool(op)?);
                            this.binary_op(mir_op, &res, &op)?
                        }
                        Op::MinMax(mmop) => {
                            if matches!(res.layout.ty.kind(), ty::Float(_)) {
                                ImmTy::from_scalar(this.fminmax_op(mmop, &res, &op)?, res.layout)
                            } else {
                                // Just boring integers, so NaNs to worry about
                                let mirop = match mmop {
                                    MinMax::Min => BinOp::Le,
                                    MinMax::Max => BinOp::Ge,
                                };
                                if this.binary_op(mirop, &res, &op)?.to_scalar().to_bool()? {
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

                let [op, init] = check_intrinsic_arg_count(args)?;
                let (op, op_len) = this.project_to_simd(op)?;
                let init = this.read_immediate(init)?;

                let mir_op = match intrinsic_name {
                    "reduce_add_ordered" => BinOp::Add,
                    "reduce_mul_ordered" => BinOp::Mul,
                    _ => unreachable!(),
                };

                let mut res = init;
                for i in 0..op_len {
                    let op = this.read_immediate(&this.project_index(&op, i)?)?;
                    res = this.binary_op(mir_op, &res, &op)?;
                }
                this.write_immediate(*res, dest)?;
            }
            "select" => {
                let [mask, yes, no] = check_intrinsic_arg_count(args)?;
                let (mask, mask_len) = this.project_to_simd(mask)?;
                let (yes, yes_len) = this.project_to_simd(yes)?;
                let (no, no_len) = this.project_to_simd(no)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, mask_len);
                assert_eq!(dest_len, yes_len);
                assert_eq!(dest_len, no_len);

                for i in 0..dest_len {
                    let mask = this.read_immediate(&this.project_index(&mask, i)?)?;
                    let yes = this.read_immediate(&this.project_index(&yes, i)?)?;
                    let no = this.read_immediate(&this.project_index(&no, i)?)?;
                    let dest = this.project_index(&dest, i)?;

                    let val = if simd_element_to_bool(mask)? { yes } else { no };
                    this.write_immediate(*val, &dest)?;
                }
            }
            // Variant of `select` that takes a bitmask rather than a "vector of bool".
            "select_bitmask" => {
                let [mask, yes, no] = check_intrinsic_arg_count(args)?;
                let (yes, yes_len) = this.project_to_simd(yes)?;
                let (no, no_len) = this.project_to_simd(no)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;
                let bitmask_len = dest_len.next_multiple_of(8);
                if bitmask_len > 64 {
                    throw_unsup_format!(
                        "simd_select_bitmask: vectors larger than 64 elements are currently not supported"
                    );
                }

                assert_eq!(dest_len, yes_len);
                assert_eq!(dest_len, no_len);

                // Read the mask, either as an integer or as an array.
                let mask: u64 = match mask.layout.ty.kind() {
                    ty::Uint(_) => {
                        // Any larger integer type is fine.
                        assert!(mask.layout.size.bits() >= bitmask_len);
                        this.read_scalar(mask)?.to_bits(mask.layout.size)?.try_into().unwrap()
                    }
                    ty::Array(elem, _len) if elem == &this.tcx.types.u8 => {
                        // The array must have exactly the right size.
                        assert_eq!(mask.layout.size.bits(), bitmask_len);
                        // Read the raw bytes.
                        let mask = mask.assert_mem_place(); // arrays cannot be immediate
                        let mask_bytes =
                            this.read_bytes_ptr_strip_provenance(mask.ptr(), mask.layout.size)?;
                        // Turn them into a `u64` in the right way.
                        let mask_size = mask.layout.size.bytes_usize();
                        let mut mask_arr = [0u8; 8];
                        match this.data_layout().endian {
                            Endian::Little => {
                                // Fill the first N bytes.
                                mask_arr[..mask_size].copy_from_slice(mask_bytes);
                                u64::from_le_bytes(mask_arr)
                            }
                            Endian::Big => {
                                // Fill the last N bytes.
                                let i = mask_arr.len().strict_sub(mask_size);
                                mask_arr[i..].copy_from_slice(mask_bytes);
                                u64::from_be_bytes(mask_arr)
                            }
                        }
                    }
                    _ => bug!("simd_select_bitmask: invalid mask type {}", mask.layout.ty),
                };

                let dest_len = u32::try_from(dest_len).unwrap();
                for i in 0..dest_len {
                    let bit_i = simd_bitmask_index(i, dest_len, this.data_layout().endian);
                    let mask = mask & 1u64.strict_shl(bit_i);
                    let yes = this.read_immediate(&this.project_index(&yes, i.into())?)?;
                    let no = this.read_immediate(&this.project_index(&no, i.into())?)?;
                    let dest = this.project_index(&dest, i.into())?;

                    let val = if mask != 0 { yes } else { no };
                    this.write_immediate(*val, &dest)?;
                }
                // The remaining bits of the mask are ignored.
            }
            // Converts a "vector of bool" into a bitmask.
            "bitmask" => {
                let [op] = check_intrinsic_arg_count(args)?;
                let (op, op_len) = this.project_to_simd(op)?;
                let bitmask_len = op_len.next_multiple_of(8);
                if bitmask_len > 64 {
                    throw_unsup_format!(
                        "simd_bitmask: vectors larger than 64 elements are currently not supported"
                    );
                }

                let op_len = u32::try_from(op_len).unwrap();
                let mut res = 0u64;
                for i in 0..op_len {
                    let op = this.read_immediate(&this.project_index(&op, i.into())?)?;
                    if simd_element_to_bool(op)? {
                        let bit_i = simd_bitmask_index(i, op_len, this.data_layout().endian);
                        res |= 1u64.strict_shl(bit_i);
                    }
                }
                // Write the result, depending on the `dest` type.
                // Returns either an unsigned integer or array of `u8`.
                match dest.layout.ty.kind() {
                    ty::Uint(_) => {
                        // Any larger integer type is fine, it will be zero-extended.
                        assert!(dest.layout.size.bits() >= bitmask_len);
                        this.write_int(res, dest)?;
                    }
                    ty::Array(elem, _len) if elem == &this.tcx.types.u8 => {
                        // The array must have exactly the right size.
                        assert_eq!(dest.layout.size.bits(), bitmask_len);
                        // We have to write the result byte-for-byte.
                        let res_size = dest.layout.size.bytes_usize();
                        let res_bytes;
                        let res_bytes_slice = match this.data_layout().endian {
                            Endian::Little => {
                                res_bytes = res.to_le_bytes();
                                &res_bytes[..res_size] // take the first N bytes
                            }
                            Endian::Big => {
                                res_bytes = res.to_be_bytes();
                                &res_bytes[res_bytes.len().strict_sub(res_size)..] // take the last N bytes
                            }
                        };
                        this.write_bytes_ptr(dest.ptr(), res_bytes_slice.iter().cloned())?;
                    }
                    _ => bug!("simd_bitmask: invalid return type {}", dest.layout.ty),
                }
            }
            "cast" | "as" | "cast_ptr" | "expose_provenance" | "with_exposed_provenance" => {
                let [op] = check_intrinsic_arg_count(args)?;
                let (op, op_len) = this.project_to_simd(op)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, op_len);

                let unsafe_cast = intrinsic_name == "cast";
                let safe_cast = intrinsic_name == "as";
                let ptr_cast = intrinsic_name == "cast_ptr";
                let expose_cast = intrinsic_name == "expose_provenance";
                let from_exposed_cast = intrinsic_name == "with_exposed_provenance";

                for i in 0..dest_len {
                    let op = this.read_immediate(&this.project_index(&op, i)?)?;
                    let dest = this.project_index(&dest, i)?;

                    let val = match (op.layout.ty.kind(), dest.layout.ty.kind()) {
                        // Int-to-(int|float): always safe
                        (ty::Int(_) | ty::Uint(_), ty::Int(_) | ty::Uint(_) | ty::Float(_))
                            if safe_cast || unsafe_cast =>
                            this.int_to_int_or_float(&op, dest.layout)?,
                        // Float-to-float: always safe
                        (ty::Float(_), ty::Float(_)) if safe_cast || unsafe_cast =>
                            this.float_to_float_or_int(&op, dest.layout)?,
                        // Float-to-int in safe mode
                        (ty::Float(_), ty::Int(_) | ty::Uint(_)) if safe_cast =>
                            this.float_to_float_or_int(&op, dest.layout)?,
                        // Float-to-int in unchecked mode
                        (ty::Float(_), ty::Int(_) | ty::Uint(_)) if unsafe_cast => {
                            this.float_to_int_checked(&op, dest.layout, Round::TowardZero)?
                                .ok_or_else(|| {
                                    err_ub_format!(
                                        "`simd_cast` intrinsic called on {op} which cannot be represented in target type `{:?}`",
                                        dest.layout.ty
                                    )
                                })?
                        }
                        // Ptr-to-ptr cast
                        (ty::RawPtr(..), ty::RawPtr(..)) if ptr_cast =>
                            this.ptr_to_ptr(&op, dest.layout)?,
                        // Ptr/Int casts
                        (ty::RawPtr(..), ty::Int(_) | ty::Uint(_)) if expose_cast =>
                            this.pointer_expose_provenance_cast(&op, dest.layout)?,
                        (ty::Int(_) | ty::Uint(_), ty::RawPtr(..)) if from_exposed_cast =>
                            this.pointer_with_exposed_provenance_cast(&op, dest.layout)?,
                        // Error otherwise
                        _ =>
                            throw_unsup_format!(
                                "Unsupported SIMD cast from element type {from_ty} to {to_ty}",
                                from_ty = op.layout.ty,
                                to_ty = dest.layout.ty,
                            ),
                    };
                    this.write_immediate(*val, &dest)?;
                }
            }
            "shuffle_const_generic" => {
                let [left, right] = check_intrinsic_arg_count(args)?;
                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                let index = generic_args[2].expect_const().to_value().valtree.unwrap_branch();
                let index_len = index.len();

                assert_eq!(left_len, right_len);
                assert_eq!(u64::try_from(index_len).unwrap(), dest_len);

                for i in 0..dest_len {
                    let src_index: u64 =
                        index[usize::try_from(i).unwrap()].unwrap_leaf().to_u32().into();
                    let dest = this.project_index(&dest, i)?;

                    let val = if src_index < left_len {
                        this.read_immediate(&this.project_index(&left, src_index)?)?
                    } else if src_index < left_len.strict_add(right_len) {
                        let right_idx = src_index.strict_sub(left_len);
                        this.read_immediate(&this.project_index(&right, right_idx)?)?
                    } else {
                        throw_ub_format!(
                            "`simd_shuffle_const_generic` index {src_index} is out-of-bounds for 2 vectors with length {dest_len}"
                        );
                    };
                    this.write_immediate(*val, &dest)?;
                }
            }
            "shuffle" => {
                let [left, right, index] = check_intrinsic_arg_count(args)?;
                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (index, index_len) = this.project_to_simd(index)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(left_len, right_len);
                assert_eq!(index_len, dest_len);

                for i in 0..dest_len {
                    let src_index: u64 = this
                        .read_immediate(&this.project_index(&index, i)?)?
                        .to_scalar()
                        .to_u32()?
                        .into();
                    let dest = this.project_index(&dest, i)?;

                    let val = if src_index < left_len {
                        this.read_immediate(&this.project_index(&left, src_index)?)?
                    } else if src_index < left_len.strict_add(right_len) {
                        let right_idx = src_index.strict_sub(left_len);
                        this.read_immediate(&this.project_index(&right, right_idx)?)?
                    } else {
                        throw_ub_format!(
                            "`simd_shuffle` index {src_index} is out-of-bounds for 2 vectors with length {dest_len}"
                        );
                    };
                    this.write_immediate(*val, &dest)?;
                }
            }
            "gather" => {
                let [passthru, ptrs, mask] = check_intrinsic_arg_count(args)?;
                let (passthru, passthru_len) = this.project_to_simd(passthru)?;
                let (ptrs, ptrs_len) = this.project_to_simd(ptrs)?;
                let (mask, mask_len) = this.project_to_simd(mask)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, passthru_len);
                assert_eq!(dest_len, ptrs_len);
                assert_eq!(dest_len, mask_len);

                for i in 0..dest_len {
                    let passthru = this.read_immediate(&this.project_index(&passthru, i)?)?;
                    let ptr = this.read_immediate(&this.project_index(&ptrs, i)?)?;
                    let mask = this.read_immediate(&this.project_index(&mask, i)?)?;
                    let dest = this.project_index(&dest, i)?;

                    let val = if simd_element_to_bool(mask)? {
                        let place = this.deref_pointer(&ptr)?;
                        this.read_immediate(&place)?
                    } else {
                        passthru
                    };
                    this.write_immediate(*val, &dest)?;
                }
            }
            "scatter" => {
                let [value, ptrs, mask] = check_intrinsic_arg_count(args)?;
                let (value, value_len) = this.project_to_simd(value)?;
                let (ptrs, ptrs_len) = this.project_to_simd(ptrs)?;
                let (mask, mask_len) = this.project_to_simd(mask)?;

                assert_eq!(ptrs_len, value_len);
                assert_eq!(ptrs_len, mask_len);

                for i in 0..ptrs_len {
                    let value = this.read_immediate(&this.project_index(&value, i)?)?;
                    let ptr = this.read_immediate(&this.project_index(&ptrs, i)?)?;
                    let mask = this.read_immediate(&this.project_index(&mask, i)?)?;

                    if simd_element_to_bool(mask)? {
                        let place = this.deref_pointer(&ptr)?;
                        this.write_immediate(*value, &place)?;
                    }
                }
            }
            "masked_load" => {
                let [mask, ptr, default] = check_intrinsic_arg_count(args)?;
                let (mask, mask_len) = this.project_to_simd(mask)?;
                let ptr = this.read_pointer(ptr)?;
                let (default, default_len) = this.project_to_simd(default)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                assert_eq!(dest_len, mask_len);
                assert_eq!(dest_len, default_len);

                for i in 0..dest_len {
                    let mask = this.read_immediate(&this.project_index(&mask, i)?)?;
                    let default = this.read_immediate(&this.project_index(&default, i)?)?;
                    let dest = this.project_index(&dest, i)?;

                    let val = if simd_element_to_bool(mask)? {
                        // Size * u64 is implemented as always checked
                        let ptr = ptr.wrapping_offset(dest.layout.size * i, this);
                        let place = this.ptr_to_mplace(ptr, dest.layout);
                        this.read_immediate(&place)?
                    } else {
                        default
                    };
                    this.write_immediate(*val, &dest)?;
                }
            }
            "masked_store" => {
                let [mask, ptr, vals] = check_intrinsic_arg_count(args)?;
                let (mask, mask_len) = this.project_to_simd(mask)?;
                let ptr = this.read_pointer(ptr)?;
                let (vals, vals_len) = this.project_to_simd(vals)?;

                assert_eq!(mask_len, vals_len);

                for i in 0..vals_len {
                    let mask = this.read_immediate(&this.project_index(&mask, i)?)?;
                    let val = this.read_immediate(&this.project_index(&vals, i)?)?;

                    if simd_element_to_bool(mask)? {
                        // Size * u64 is implemented as always checked
                        let ptr = ptr.wrapping_offset(val.layout.size * i, this);
                        let place = this.ptr_to_mplace(ptr, val.layout);
                        this.write_immediate(*val, &place)?
                    };
                }
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }

    fn fminmax_op(
        &self,
        op: MinMax,
        left: &ImmTy<'tcx>,
        right: &ImmTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_ref();
        assert_eq!(left.layout.ty, right.layout.ty);
        let ty::Float(float_ty) = left.layout.ty.kind() else {
            bug!("fmax operand is not a float")
        };
        let left = left.to_scalar();
        let right = right.to_scalar();
        interp_ok(match float_ty {
            FloatTy::F16 => unimplemented!("f16_f128"),
            FloatTy::F32 => {
                let left = left.to_f32()?;
                let right = right.to_f32()?;
                let res = match op {
                    MinMax::Min => left.min(right),
                    MinMax::Max => left.max(right),
                };
                let res = this.adjust_nan(res, &[left, right]);
                Scalar::from_f32(res)
            }
            FloatTy::F64 => {
                let left = left.to_f64()?;
                let right = right.to_f64()?;
                let res = match op {
                    MinMax::Min => left.min(right),
                    MinMax::Max => left.max(right),
                };
                let res = this.adjust_nan(res, &[left, right]);
                Scalar::from_f64(res)
            }
            FloatTy::F128 => unimplemented!("f16_f128"),
        })
    }
}

fn simd_bitmask_index(idx: u32, vec_len: u32, endianness: Endian) -> u32 {
    assert!(idx < vec_len);
    match endianness {
        Endian::Little => idx,
        #[expect(clippy::arithmetic_side_effects)] // idx < vec_len
        Endian::Big => vec_len - 1 - idx, // reverse order of bits
    }
}
