use either::Either;
use rustc_abi::{BackendRepr, Endian};
use rustc_apfloat::ieee::{Double, Half, Quad, Single};
use rustc_apfloat::{Float, Round};
use rustc_middle::mir::interpret::{InterpErrorKind, Pointer, UndefinedBehaviorInfo};
use rustc_middle::ty::{FloatTy, ScalarInt, SimdAlign};
use rustc_middle::{bug, err_ub_format, mir, span_bug, throw_unsup_format, ty};
use rustc_span::{Symbol, sym};
use tracing::trace;

use super::{
    ImmTy, InterpCx, InterpResult, Machine, MinMax, MulAddType, OpTy, PlaceTy, Provenance, Scalar,
    Size, TyAndLayout, assert_matches, interp_ok, throw_ub_format,
};
use crate::interpret::Writeable;

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    /// Returns `true` if emulation happened.
    /// Here we implement the intrinsics that are common to all CTFE instances; individual machines can add their own
    /// intrinsic handling.
    pub fn eval_simd_intrinsic(
        &mut self,
        intrinsic_name: Symbol,
        generic_args: ty::GenericArgsRef<'tcx>,
        args: &[OpTy<'tcx, M::Provenance>],
        dest: &PlaceTy<'tcx, M::Provenance>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, bool> {
        let dest = dest.force_mplace(self)?;

        match intrinsic_name {
            sym::simd_insert => {
                let index = u64::from(self.read_scalar(&args[1])?.to_u32()?);
                let elem = &args[2];
                let (input, input_len) = self.project_to_simd(&args[0])?;
                let (dest, dest_len) = self.project_to_simd(&dest)?;
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
                self.copy_op(&self.project_index(&input, index)?, &dest)?;
            }
            sym::simd_neg
            | sym::simd_fabs
            | sym::simd_ceil
            | sym::simd_floor
            | sym::simd_round
            | sym::simd_round_ties_even
            | sym::simd_trunc
            | sym::simd_ctlz
            | sym::simd_ctpop
            | sym::simd_cttz
            | sym::simd_bswap
            | sym::simd_bitreverse => {
                let (op, op_len) = self.project_to_simd(&args[0])?;
                let (dest, dest_len) = self.project_to_simd(&dest)?;

                assert_eq!(dest_len, op_len);

                #[derive(Copy, Clone)]
                enum Op {
                    MirOp(mir::UnOp),
                    Abs,
                    Round(rustc_apfloat::Round),
                    Numeric(Symbol),
                }
                let which = match intrinsic_name {
                    sym::simd_neg => Op::MirOp(mir::UnOp::Neg),
                    sym::simd_fabs => Op::Abs,
                    sym::simd_ceil => Op::Round(rustc_apfloat::Round::TowardPositive),
                    sym::simd_floor => Op::Round(rustc_apfloat::Round::TowardNegative),
                    sym::simd_round => Op::Round(rustc_apfloat::Round::NearestTiesToAway),
                    sym::simd_round_ties_even => Op::Round(rustc_apfloat::Round::NearestTiesToEven),
                    sym::simd_trunc => Op::Round(rustc_apfloat::Round::TowardZero),
                    sym::simd_ctlz => Op::Numeric(sym::ctlz),
                    sym::simd_ctpop => Op::Numeric(sym::ctpop),
                    sym::simd_cttz => Op::Numeric(sym::cttz),
                    sym::simd_bswap => Op::Numeric(sym::bswap),
                    sym::simd_bitreverse => Op::Numeric(sym::bitreverse),
                    _ => unreachable!(),
                };

                for i in 0..dest_len {
                    let op = self.read_immediate(&self.project_index(&op, i)?)?;
                    let dest = self.project_index(&dest, i)?;
                    let val = match which {
                        Op::MirOp(mir_op) => {
                            // this already does NaN adjustments
                            self.unary_op(mir_op, &op)?.to_scalar()
                        }
                        Op::Abs => {
                            // Works for f32 and f64.
                            let ty::Float(float_ty) = op.layout.ty.kind() else {
                                span_bug!(
                                    self.cur_span(),
                                    "{} operand is not a float",
                                    intrinsic_name
                                )
                            };
                            let op = op.to_scalar();
                            // "Bitwise" operation, no NaN adjustments
                            match float_ty {
                                FloatTy::F16 => Scalar::from_f16(op.to_f16()?.abs()),
                                FloatTy::F32 => Scalar::from_f32(op.to_f32()?.abs()),
                                FloatTy::F64 => Scalar::from_f64(op.to_f64()?.abs()),
                                FloatTy::F128 => Scalar::from_f128(op.to_f128()?.abs()),
                            }
                        }
                        Op::Round(rounding) => {
                            let ty::Float(float_ty) = op.layout.ty.kind() else {
                                span_bug!(
                                    self.cur_span(),
                                    "{} operand is not a float",
                                    intrinsic_name
                                )
                            };
                            let op = op.to_scalar();
                            match float_ty {
                                FloatTy::F16 => self.float_round::<Half>(op, rounding)?,
                                FloatTy::F32 => self.float_round::<Single>(op, rounding)?,
                                FloatTy::F64 => self.float_round::<Double>(op, rounding)?,
                                FloatTy::F128 => self.float_round::<Quad>(op, rounding)?,
                            }
                        }
                        Op::Numeric(name) => {
                            self.numeric_intrinsic(name, op.to_scalar(), op.layout, op.layout)?
                        }
                    };
                    self.write_scalar(val, &dest)?;
                }
            }
            sym::simd_add
            | sym::simd_sub
            | sym::simd_mul
            | sym::simd_div
            | sym::simd_rem
            | sym::simd_shl
            | sym::simd_shr
            | sym::simd_and
            | sym::simd_or
            | sym::simd_xor
            | sym::simd_eq
            | sym::simd_ne
            | sym::simd_lt
            | sym::simd_le
            | sym::simd_gt
            | sym::simd_ge
            | sym::simd_fmax
            | sym::simd_fmin
            | sym::simd_saturating_add
            | sym::simd_saturating_sub
            | sym::simd_arith_offset => {
                use mir::BinOp;

                let (left, left_len) = self.project_to_simd(&args[0])?;
                let (right, right_len) = self.project_to_simd(&args[1])?;
                let (dest, dest_len) = self.project_to_simd(&dest)?;

                assert_eq!(dest_len, left_len);
                assert_eq!(dest_len, right_len);

                enum Op {
                    MirOp(BinOp),
                    SaturatingOp(BinOp),
                    FMinMax(MinMax),
                    WrappingOffset,
                }
                let which = match intrinsic_name {
                    sym::simd_add => Op::MirOp(BinOp::Add),
                    sym::simd_sub => Op::MirOp(BinOp::Sub),
                    sym::simd_mul => Op::MirOp(BinOp::Mul),
                    sym::simd_div => Op::MirOp(BinOp::Div),
                    sym::simd_rem => Op::MirOp(BinOp::Rem),
                    sym::simd_shl => Op::MirOp(BinOp::ShlUnchecked),
                    sym::simd_shr => Op::MirOp(BinOp::ShrUnchecked),
                    sym::simd_and => Op::MirOp(BinOp::BitAnd),
                    sym::simd_or => Op::MirOp(BinOp::BitOr),
                    sym::simd_xor => Op::MirOp(BinOp::BitXor),
                    sym::simd_eq => Op::MirOp(BinOp::Eq),
                    sym::simd_ne => Op::MirOp(BinOp::Ne),
                    sym::simd_lt => Op::MirOp(BinOp::Lt),
                    sym::simd_le => Op::MirOp(BinOp::Le),
                    sym::simd_gt => Op::MirOp(BinOp::Gt),
                    sym::simd_ge => Op::MirOp(BinOp::Ge),
                    sym::simd_fmax => Op::FMinMax(MinMax::MaxNum),
                    sym::simd_fmin => Op::FMinMax(MinMax::MinNum),
                    sym::simd_saturating_add => Op::SaturatingOp(BinOp::Add),
                    sym::simd_saturating_sub => Op::SaturatingOp(BinOp::Sub),
                    sym::simd_arith_offset => Op::WrappingOffset,
                    _ => unreachable!(),
                };

                for i in 0..dest_len {
                    let left = self.read_immediate(&self.project_index(&left, i)?)?;
                    let right = self.read_immediate(&self.project_index(&right, i)?)?;
                    let dest = self.project_index(&dest, i)?;
                    let val = match which {
                        Op::MirOp(mir_op) => {
                            // this does NaN adjustments.
                            let val = self.binary_op(mir_op, &left, &right).map_err_kind(|kind| {
                                match kind {
                                    InterpErrorKind::UndefinedBehavior(UndefinedBehaviorInfo::ShiftOverflow { shift_amount, .. }) => {
                                        // this resets the interpreter backtrace, but it's not worth avoiding that.
                                        let shift_amount = match shift_amount {
                                            Either::Left(v) => v.to_string(),
                                            Either::Right(v) => v.to_string(),
                                        };
                                        err_ub_format!("overflowing shift by {shift_amount} in `{intrinsic_name}` in lane {i}")
                                    }
                                    kind => kind
                                }
                            })?;
                            if matches!(
                                mir_op,
                                BinOp::Eq
                                    | BinOp::Ne
                                    | BinOp::Lt
                                    | BinOp::Le
                                    | BinOp::Gt
                                    | BinOp::Ge
                            ) {
                                // Special handling for boolean-returning operations
                                assert_eq!(val.layout.ty, self.tcx.types.bool);
                                let val = val.to_scalar().to_bool().unwrap();
                                bool_to_simd_element(val, dest.layout.size)
                            } else {
                                assert_ne!(val.layout.ty, self.tcx.types.bool);
                                assert_eq!(val.layout.ty, dest.layout.ty);
                                val.to_scalar()
                            }
                        }
                        Op::SaturatingOp(mir_op) => self.saturating_arith(mir_op, &left, &right)?,
                        Op::WrappingOffset => {
                            let ptr = left.to_scalar().to_pointer(self)?;
                            let offset_count = right.to_scalar().to_target_isize(self)?;
                            let pointee_ty = left.layout.ty.builtin_deref(true).unwrap();

                            let pointee_size =
                                i64::try_from(self.layout_of(pointee_ty)?.size.bytes()).unwrap();
                            let offset_bytes = offset_count.wrapping_mul(pointee_size);
                            let offset_ptr = ptr.wrapping_signed_offset(offset_bytes, self);
                            Scalar::from_maybe_pointer(offset_ptr, self)
                        }
                        Op::FMinMax(op) => self.fminmax_op(op, &left, &right)?,
                    };
                    self.write_scalar(val, &dest)?;
                }
            }
            sym::simd_reduce_and
            | sym::simd_reduce_or
            | sym::simd_reduce_xor
            | sym::simd_reduce_any
            | sym::simd_reduce_all
            | sym::simd_reduce_max
            | sym::simd_reduce_min => {
                use mir::BinOp;

                let (op, op_len) = self.project_to_simd(&args[0])?;

                let imm_from_bool = |b| {
                    ImmTy::from_scalar(
                        Scalar::from_bool(b),
                        self.layout_of(self.tcx.types.bool).unwrap(),
                    )
                };

                enum Op {
                    MirOp(BinOp),
                    MirOpBool(BinOp),
                    MinMax(MinMax),
                }
                let which = match intrinsic_name {
                    sym::simd_reduce_and => Op::MirOp(BinOp::BitAnd),
                    sym::simd_reduce_or => Op::MirOp(BinOp::BitOr),
                    sym::simd_reduce_xor => Op::MirOp(BinOp::BitXor),
                    sym::simd_reduce_any => Op::MirOpBool(BinOp::BitOr),
                    sym::simd_reduce_all => Op::MirOpBool(BinOp::BitAnd),
                    sym::simd_reduce_max => Op::MinMax(MinMax::MaxNum),
                    sym::simd_reduce_min => Op::MinMax(MinMax::MinNum),
                    _ => unreachable!(),
                };

                // Initialize with first lane, then proceed with the rest.
                let mut res = self.read_immediate(&self.project_index(&op, 0)?)?;
                if matches!(which, Op::MirOpBool(_)) {
                    // Convert to `bool` scalar.
                    res = imm_from_bool(simd_element_to_bool(res)?);
                }
                for i in 1..op_len {
                    let op = self.read_immediate(&self.project_index(&op, i)?)?;
                    res = match which {
                        Op::MirOp(mir_op) => self.binary_op(mir_op, &res, &op)?,
                        Op::MirOpBool(mir_op) => {
                            let op = imm_from_bool(simd_element_to_bool(op)?);
                            self.binary_op(mir_op, &res, &op)?
                        }
                        Op::MinMax(mmop) => {
                            if matches!(res.layout.ty.kind(), ty::Float(_)) {
                                ImmTy::from_scalar(self.fminmax_op(mmop, &res, &op)?, res.layout)
                            } else {
                                // Just boring integers, no NaNs to worry about.
                                let mirop = match mmop {
                                    MinMax::MinNum | MinMax::Minimum => BinOp::Le,
                                    MinMax::MaxNum | MinMax::Maximum => BinOp::Ge,
                                };
                                if self.binary_op(mirop, &res, &op)?.to_scalar().to_bool()? {
                                    res
                                } else {
                                    op
                                }
                            }
                        }
                    };
                }
                self.write_immediate(*res, &dest)?;
            }
            sym::simd_reduce_add_ordered | sym::simd_reduce_mul_ordered => {
                use mir::BinOp;

                let (op, op_len) = self.project_to_simd(&args[0])?;
                let init = self.read_immediate(&args[1])?;

                let mir_op = match intrinsic_name {
                    sym::simd_reduce_add_ordered => BinOp::Add,
                    sym::simd_reduce_mul_ordered => BinOp::Mul,
                    _ => unreachable!(),
                };

                let mut res = init;
                for i in 0..op_len {
                    let op = self.read_immediate(&self.project_index(&op, i)?)?;
                    res = self.binary_op(mir_op, &res, &op)?;
                }
                self.write_immediate(*res, &dest)?;
            }
            sym::simd_select => {
                let (mask, mask_len) = self.project_to_simd(&args[0])?;
                let (yes, yes_len) = self.project_to_simd(&args[1])?;
                let (no, no_len) = self.project_to_simd(&args[2])?;
                let (dest, dest_len) = self.project_to_simd(&dest)?;

                assert_eq!(dest_len, mask_len);
                assert_eq!(dest_len, yes_len);
                assert_eq!(dest_len, no_len);

                for i in 0..dest_len {
                    let mask = self.read_immediate(&self.project_index(&mask, i)?)?;
                    let yes = self.read_immediate(&self.project_index(&yes, i)?)?;
                    let no = self.read_immediate(&self.project_index(&no, i)?)?;
                    let dest = self.project_index(&dest, i)?;

                    let val = if simd_element_to_bool(mask)? { yes } else { no };
                    self.write_immediate(*val, &dest)?;
                }
            }
            // Variant of `select` that takes a bitmask rather than a "vector of bool".
            sym::simd_select_bitmask => {
                let mask = &args[0];
                let (yes, yes_len) = self.project_to_simd(&args[1])?;
                let (no, no_len) = self.project_to_simd(&args[2])?;
                let (dest, dest_len) = self.project_to_simd(&dest)?;
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
                        self.read_scalar(mask)?.to_bits(mask.layout.size)?.try_into().unwrap()
                    }
                    ty::Array(elem, _len) if elem == &self.tcx.types.u8 => {
                        // The array must have exactly the right size.
                        assert_eq!(mask.layout.size.bits(), bitmask_len);
                        // Read the raw bytes.
                        let mask = mask.assert_mem_place(); // arrays cannot be immediate
                        let mask_bytes =
                            self.read_bytes_ptr_strip_provenance(mask.ptr(), mask.layout.size)?;
                        // Turn them into a `u64` in the right way.
                        let mask_size = mask.layout.size.bytes_usize();
                        let mut mask_arr = [0u8; 8];
                        match self.tcx.data_layout.endian {
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
                    let bit_i = simd_bitmask_index(i, dest_len, self.tcx.data_layout.endian);
                    let mask = mask & 1u64.strict_shl(bit_i);
                    let yes = self.read_immediate(&self.project_index(&yes, i.into())?)?;
                    let no = self.read_immediate(&self.project_index(&no, i.into())?)?;
                    let dest = self.project_index(&dest, i.into())?;

                    let val = if mask != 0 { yes } else { no };
                    self.write_immediate(*val, &dest)?;
                }
                // The remaining bits of the mask are ignored.
            }
            // Converts a "vector of bool" into a bitmask.
            sym::simd_bitmask => {
                let (op, op_len) = self.project_to_simd(&args[0])?;
                let bitmask_len = op_len.next_multiple_of(8);
                if bitmask_len > 64 {
                    throw_unsup_format!(
                        "simd_bitmask: vectors larger than 64 elements are currently not supported"
                    );
                }

                let op_len = u32::try_from(op_len).unwrap();
                let mut res = 0u64;
                for i in 0..op_len {
                    let op = self.read_immediate(&self.project_index(&op, i.into())?)?;
                    if simd_element_to_bool(op)? {
                        let bit_i = simd_bitmask_index(i, op_len, self.tcx.data_layout.endian);
                        res |= 1u64.strict_shl(bit_i);
                    }
                }
                // Write the result, depending on the `dest` type.
                // Returns either an unsigned integer or array of `u8`.
                match dest.layout.ty.kind() {
                    ty::Uint(_) => {
                        // Any larger integer type is fine, it will be zero-extended.
                        assert!(dest.layout.size.bits() >= bitmask_len);
                        self.write_scalar(Scalar::from_uint(res, dest.layout.size), &dest)?;
                    }
                    ty::Array(elem, _len) if elem == &self.tcx.types.u8 => {
                        // The array must have exactly the right size.
                        assert_eq!(dest.layout.size.bits(), bitmask_len);
                        // We have to write the result byte-for-byte.
                        let res_size = dest.layout.size.bytes_usize();
                        let res_bytes;
                        let res_bytes_slice = match self.tcx.data_layout.endian {
                            Endian::Little => {
                                res_bytes = res.to_le_bytes();
                                &res_bytes[..res_size] // take the first N bytes
                            }
                            Endian::Big => {
                                res_bytes = res.to_be_bytes();
                                &res_bytes[res_bytes.len().strict_sub(res_size)..] // take the last N bytes
                            }
                        };
                        self.write_bytes_ptr(dest.ptr(), res_bytes_slice.iter().cloned())?;
                    }
                    _ => bug!("simd_bitmask: invalid return type {}", dest.layout.ty),
                }
            }
            sym::simd_cast
            | sym::simd_as
            | sym::simd_cast_ptr
            | sym::simd_with_exposed_provenance => {
                let (op, op_len) = self.project_to_simd(&args[0])?;
                let (dest, dest_len) = self.project_to_simd(&dest)?;

                assert_eq!(dest_len, op_len);

                let unsafe_cast = intrinsic_name == sym::simd_cast;
                let safe_cast = intrinsic_name == sym::simd_as;
                let ptr_cast = intrinsic_name == sym::simd_cast_ptr;
                let from_exposed_cast = intrinsic_name == sym::simd_with_exposed_provenance;

                for i in 0..dest_len {
                    let op = self.read_immediate(&self.project_index(&op, i)?)?;
                    let dest = self.project_index(&dest, i)?;

                    let val = match (op.layout.ty.kind(), dest.layout.ty.kind()) {
                        // Int-to-(int|float): always safe
                        (ty::Int(_) | ty::Uint(_), ty::Int(_) | ty::Uint(_) | ty::Float(_))
                            if safe_cast || unsafe_cast =>
                            self.int_to_int_or_float(&op, dest.layout)?,
                        // Float-to-float: always safe
                        (ty::Float(_), ty::Float(_)) if safe_cast || unsafe_cast =>
                            self.float_to_float_or_int(&op, dest.layout)?,
                        // Float-to-int in safe mode
                        (ty::Float(_), ty::Int(_) | ty::Uint(_)) if safe_cast =>
                            self.float_to_float_or_int(&op, dest.layout)?,
                        // Float-to-int in unchecked mode
                        (ty::Float(_), ty::Int(_) | ty::Uint(_)) if unsafe_cast => {
                            self.float_to_int_checked(&op, dest.layout, Round::TowardZero)?
                                .ok_or_else(|| {
                                    err_ub_format!(
                                        "`simd_cast` intrinsic called on {op} which cannot be represented in target type `{:?}`",
                                        dest.layout.ty
                                    )
                                })?
                        }
                        // Ptr-to-ptr cast
                        (ty::RawPtr(..), ty::RawPtr(..)) if ptr_cast =>
                            self.ptr_to_ptr(&op, dest.layout)?,
                        // Int->Ptr casts
                        (ty::Int(_) | ty::Uint(_), ty::RawPtr(..)) if from_exposed_cast =>
                            self.pointer_with_exposed_provenance_cast(&op, dest.layout)?,
                        // Error otherwise
                        _ =>
                            throw_unsup_format!(
                                "Unsupported SIMD cast from element type {from_ty} to {to_ty}",
                                from_ty = op.layout.ty,
                                to_ty = dest.layout.ty,
                            ),
                    };
                    self.write_immediate(*val, &dest)?;
                }
            }
            sym::simd_shuffle_const_generic => {
                let (left, left_len) = self.project_to_simd(&args[0])?;
                let (right, right_len) = self.project_to_simd(&args[1])?;
                let (dest, dest_len) = self.project_to_simd(&dest)?;

                let index = generic_args[2].expect_const().to_value().valtree.unwrap_branch();
                let index_len = index.len();

                assert_eq!(left_len, right_len);
                assert_eq!(u64::try_from(index_len).unwrap(), dest_len);

                for i in 0..dest_len {
                    let src_index: u64 =
                        index[usize::try_from(i).unwrap()].unwrap_leaf().to_u32().into();
                    let dest = self.project_index(&dest, i)?;

                    let val = if src_index < left_len {
                        self.read_immediate(&self.project_index(&left, src_index)?)?
                    } else if src_index < left_len.strict_add(right_len) {
                        let right_idx = src_index.strict_sub(left_len);
                        self.read_immediate(&self.project_index(&right, right_idx)?)?
                    } else {
                        throw_ub_format!(
                            "`simd_shuffle_const_generic` index {src_index} is out-of-bounds for 2 vectors with length {dest_len}"
                        );
                    };
                    self.write_immediate(*val, &dest)?;
                }
            }
            sym::simd_shuffle => {
                let (left, left_len) = self.project_to_simd(&args[0])?;
                let (right, right_len) = self.project_to_simd(&args[1])?;
                let (index, index_len) = self.project_to_simd(&args[2])?;
                let (dest, dest_len) = self.project_to_simd(&dest)?;

                assert_eq!(left_len, right_len);
                assert_eq!(index_len, dest_len);

                for i in 0..dest_len {
                    let src_index: u64 = self
                        .read_immediate(&self.project_index(&index, i)?)?
                        .to_scalar()
                        .to_u32()?
                        .into();
                    let dest = self.project_index(&dest, i)?;

                    let val = if src_index < left_len {
                        self.read_immediate(&self.project_index(&left, src_index)?)?
                    } else if src_index < left_len.strict_add(right_len) {
                        let right_idx = src_index.strict_sub(left_len);
                        self.read_immediate(&self.project_index(&right, right_idx)?)?
                    } else {
                        throw_ub_format!(
                            "`simd_shuffle` index {src_index} is out-of-bounds for 2 vectors with length {dest_len}"
                        );
                    };
                    self.write_immediate(*val, &dest)?;
                }
            }
            sym::simd_gather => {
                let (passthru, passthru_len) = self.project_to_simd(&args[0])?;
                let (ptrs, ptrs_len) = self.project_to_simd(&args[1])?;
                let (mask, mask_len) = self.project_to_simd(&args[2])?;
                let (dest, dest_len) = self.project_to_simd(&dest)?;

                assert_eq!(dest_len, passthru_len);
                assert_eq!(dest_len, ptrs_len);
                assert_eq!(dest_len, mask_len);

                for i in 0..dest_len {
                    let passthru = self.read_immediate(&self.project_index(&passthru, i)?)?;
                    let ptr = self.read_immediate(&self.project_index(&ptrs, i)?)?;
                    let mask = self.read_immediate(&self.project_index(&mask, i)?)?;
                    let dest = self.project_index(&dest, i)?;

                    let val = if simd_element_to_bool(mask)? {
                        let place = self.deref_pointer(&ptr)?;
                        self.read_immediate(&place)?
                    } else {
                        passthru
                    };
                    self.write_immediate(*val, &dest)?;
                }
            }
            sym::simd_scatter => {
                let (value, value_len) = self.project_to_simd(&args[0])?;
                let (ptrs, ptrs_len) = self.project_to_simd(&args[1])?;
                let (mask, mask_len) = self.project_to_simd(&args[2])?;

                assert_eq!(ptrs_len, value_len);
                assert_eq!(ptrs_len, mask_len);

                for i in 0..ptrs_len {
                    let value = self.read_immediate(&self.project_index(&value, i)?)?;
                    let ptr = self.read_immediate(&self.project_index(&ptrs, i)?)?;
                    let mask = self.read_immediate(&self.project_index(&mask, i)?)?;

                    if simd_element_to_bool(mask)? {
                        let place = self.deref_pointer(&ptr)?;
                        self.write_immediate(*value, &place)?;
                    }
                }
            }
            sym::simd_masked_load => {
                let dest_layout = dest.layout;

                let (mask, mask_len) = self.project_to_simd(&args[0])?;
                let ptr = self.read_pointer(&args[1])?;
                let (default, default_len) = self.project_to_simd(&args[2])?;
                let (dest, dest_len) = self.project_to_simd(&dest)?;

                assert_eq!(dest_len, mask_len);
                assert_eq!(dest_len, default_len);

                self.check_simd_ptr_alignment(
                    ptr,
                    dest_layout,
                    generic_args[3].expect_const().to_value().valtree.unwrap_branch()[0]
                        .unwrap_leaf()
                        .to_simd_alignment(),
                )?;

                for i in 0..dest_len {
                    let mask = self.read_immediate(&self.project_index(&mask, i)?)?;
                    let default = self.read_immediate(&self.project_index(&default, i)?)?;
                    let dest = self.project_index(&dest, i)?;

                    let val = if simd_element_to_bool(mask)? {
                        // Size * u64 is implemented as always checked
                        let ptr = ptr.wrapping_offset(dest.layout.size * i, self);
                        // we have already checked the alignment requirements
                        let place = self.ptr_to_mplace_unaligned(ptr, dest.layout);
                        self.read_immediate(&place)?
                    } else {
                        default
                    };
                    self.write_immediate(*val, &dest)?;
                }
            }
            sym::simd_masked_store => {
                let (mask, mask_len) = self.project_to_simd(&args[0])?;
                let ptr = self.read_pointer(&args[1])?;
                let (vals, vals_len) = self.project_to_simd(&args[2])?;

                assert_eq!(mask_len, vals_len);

                self.check_simd_ptr_alignment(
                    ptr,
                    args[2].layout,
                    generic_args[3].expect_const().to_value().valtree.unwrap_branch()[0]
                        .unwrap_leaf()
                        .to_simd_alignment(),
                )?;

                for i in 0..vals_len {
                    let mask = self.read_immediate(&self.project_index(&mask, i)?)?;
                    let val = self.read_immediate(&self.project_index(&vals, i)?)?;

                    if simd_element_to_bool(mask)? {
                        // Size * u64 is implemented as always checked
                        let ptr = ptr.wrapping_offset(val.layout.size * i, self);
                        // we have already checked the alignment requirements
                        let place = self.ptr_to_mplace_unaligned(ptr, val.layout);
                        self.write_immediate(*val, &place)?
                    };
                }
            }
            sym::simd_fma | sym::simd_relaxed_fma => {
                // `simd_fma` should always deterministically use `mul_add`, whereas `relaxed_fma`
                // is non-deterministic, and can use either `mul_add` or `a * b + c`
                let typ = match intrinsic_name {
                    sym::simd_fma => MulAddType::Fused,
                    sym::simd_relaxed_fma => MulAddType::Nondeterministic,
                    _ => unreachable!(),
                };

                let (a, a_len) = self.project_to_simd(&args[0])?;
                let (b, b_len) = self.project_to_simd(&args[1])?;
                let (c, c_len) = self.project_to_simd(&args[2])?;
                let (dest, dest_len) = self.project_to_simd(&dest)?;

                assert_eq!(dest_len, a_len);
                assert_eq!(dest_len, b_len);
                assert_eq!(dest_len, c_len);

                for i in 0..dest_len {
                    let a = self.read_scalar(&self.project_index(&a, i)?)?;
                    let b = self.read_scalar(&self.project_index(&b, i)?)?;
                    let c = self.read_scalar(&self.project_index(&c, i)?)?;
                    let dest = self.project_index(&dest, i)?;

                    let ty::Float(float_ty) = dest.layout.ty.kind() else {
                        span_bug!(self.cur_span(), "{} operand is not a float", intrinsic_name)
                    };

                    let val = match float_ty {
                        FloatTy::F16 => self.float_muladd::<Half>(a, b, c, typ)?,
                        FloatTy::F32 => self.float_muladd::<Single>(a, b, c, typ)?,
                        FloatTy::F64 => self.float_muladd::<Double>(a, b, c, typ)?,
                        FloatTy::F128 => self.float_muladd::<Quad>(a, b, c, typ)?,
                    };
                    self.write_scalar(val, &dest)?;
                }
            }
            sym::simd_funnel_shl | sym::simd_funnel_shr => {
                let (left, _) = self.project_to_simd(&args[0])?;
                let (right, _) = self.project_to_simd(&args[1])?;
                let (shift, _) = self.project_to_simd(&args[2])?;
                let (dest, _) = self.project_to_simd(&dest)?;

                let (len, elem_ty) = args[0].layout.ty.simd_size_and_type(*self.tcx);
                let (elem_size, _signed) = elem_ty.int_size_and_signed(*self.tcx);
                let elem_size_bits = u128::from(elem_size.bits());

                let is_left = intrinsic_name == sym::simd_funnel_shl;

                for i in 0..len {
                    let left =
                        self.read_scalar(&self.project_index(&left, i)?)?.to_bits(elem_size)?;
                    let right =
                        self.read_scalar(&self.project_index(&right, i)?)?.to_bits(elem_size)?;
                    let shift_bits =
                        self.read_scalar(&self.project_index(&shift, i)?)?.to_bits(elem_size)?;

                    if shift_bits >= elem_size_bits {
                        throw_ub_format!(
                            "overflowing shift by {shift_bits} in `{intrinsic_name}` in lane {i}"
                        );
                    }
                    let inv_shift_bits = u32::try_from(elem_size_bits - shift_bits).unwrap();

                    // A funnel shift left by S can be implemented as `(x << S) | y.unbounded_shr(SIZE - S)`.
                    // The `unbounded_shr` is needed because otherwise if `S = 0`, it would be `x | y`
                    // when it should be `x`.
                    //
                    // This selects the least-significant `SIZE - S` bits of `x`, followed by the `S` most
                    // significant bits of `y`. As `left` and `right` both occupy the lower `SIZE` bits,
                    // we can treat the lower `SIZE` bits as an integer of the right width and use
                    // the same implementation, but on a zero-extended `x` and `y`. This works because
                    // `x << S` just pushes the `SIZE-S` MSBs out, and `y >> (SIZE - S)` shifts in
                    // zeros, as it is zero-extended. To the lower `SIZE` bits, this looks just like a
                    // funnel shift left.
                    //
                    // Note that the `unbounded_sh{l,r}`s are needed only in case we are using this on
                    // `u128xN` and `inv_shift_bits == 128`.
                    let result_bits = if is_left {
                        (left << shift_bits) | right.unbounded_shr(inv_shift_bits)
                    } else {
                        left.unbounded_shl(inv_shift_bits) | (right >> shift_bits)
                    };
                    let (result, _overflow) = ScalarInt::truncate_from_uint(result_bits, elem_size);

                    let dest = self.project_index(&dest, i)?;
                    self.write_scalar(result, &dest)?;
                }
            }

            // Unsupported intrinsic: skip the return_to_block below.
            _ => return interp_ok(false),
        }

        trace!("{:?}", self.dump_place(&dest.clone().into()));
        self.return_to_block(ret)?;
        interp_ok(true)
    }

    fn fminmax_op(
        &self,
        op: MinMax,
        left: &ImmTy<'tcx, M::Provenance>,
        right: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>> {
        assert_eq!(left.layout.ty, right.layout.ty);
        let ty::Float(float_ty) = left.layout.ty.kind() else {
            bug!("fmax operand is not a float")
        };
        let left = left.to_scalar();
        let right = right.to_scalar();
        interp_ok(match float_ty {
            FloatTy::F16 => self.float_minmax::<Half>(left, right, op)?,
            FloatTy::F32 => self.float_minmax::<Single>(left, right, op)?,
            FloatTy::F64 => self.float_minmax::<Double>(left, right, op)?,
            FloatTy::F128 => self.float_minmax::<Quad>(left, right, op)?,
        })
    }

    fn check_simd_ptr_alignment(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        vector_layout: TyAndLayout<'tcx>,
        alignment: SimdAlign,
    ) -> InterpResult<'tcx> {
        assert_matches!(vector_layout.backend_repr, BackendRepr::SimdVector { .. });

        let align = match alignment {
            ty::SimdAlign::Unaligned => {
                // The pointer is supposed to be unaligned, so no check is required.
                return interp_ok(());
            }
            ty::SimdAlign::Element => {
                // Take the alignment of the only field, which is an array and therefore has the same
                // alignment as the element type.
                vector_layout.field(self, 0).align.abi
            }
            ty::SimdAlign::Vector => vector_layout.align.abi,
        };

        self.check_ptr_align(ptr, align)
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

fn bool_to_simd_element<Prov: Provenance>(b: bool, size: Size) -> Scalar<Prov> {
    // SIMD uses all-1 as pattern for "true". In two's complement,
    // -1 has all its bits set to one and `from_int` will truncate or
    // sign-extend it to `size` as required.
    let val = if b { -1 } else { 0 };
    Scalar::from_int(val, size)
}

fn simd_element_to_bool<Prov: Provenance>(elem: ImmTy<'_, Prov>) -> InterpResult<'_, bool> {
    assert!(
        matches!(elem.layout.ty.kind(), ty::Int(_) | ty::Uint(_)),
        "SIMD mask element type must be an integer, but this is `{}`",
        elem.layout.ty
    );
    let val = elem.to_scalar().to_int(elem.layout.size)?;
    interp_ok(match val {
        0 => false,
        -1 => true,
        _ => throw_ub_format!("each element of a SIMD mask must be all-0-bits or all-1-bits"),
    })
}
