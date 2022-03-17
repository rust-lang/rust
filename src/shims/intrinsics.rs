use std::convert::TryInto;
use std::iter;

use log::trace;

use rustc_apfloat::{Float, Round};
use rustc_middle::ty::layout::{HasParamEnv, IntegerExt, LayoutOf};
use rustc_middle::{mir, mir::BinOp, ty, ty::FloatTy};
use rustc_target::abi::{Align, Endian, HasDataLayout, Integer, Size};

use crate::*;
use helpers::check_arg_count;

pub enum AtomicOp {
    MirOp(mir::BinOp, bool),
    Max,
    Min,
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn call_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(&PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
        _unwind: StackPopUnwind,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        if this.emulate_intrinsic(instance, args, ret)? {
            return Ok(());
        }

        // All supported intrinsics have a return place.
        let intrinsic_name = this.tcx.item_name(instance.def_id());
        let intrinsic_name = intrinsic_name.as_str();
        let (dest, ret) = match ret {
            None => throw_unsup_format!("unimplemented (diverging) intrinsic: {}", intrinsic_name),
            Some(p) => p,
        };

        // Then handle terminating intrinsics.
        match intrinsic_name {
            // Miri overwriting CTFE intrinsics.
            "ptr_guaranteed_eq" => {
                let &[ref left, ref right] = check_arg_count(args)?;
                let left = this.read_immediate(left)?;
                let right = this.read_immediate(right)?;
                this.binop_ignore_overflow(mir::BinOp::Eq, &left, &right, dest)?;
            }
            "ptr_guaranteed_ne" => {
                let &[ref left, ref right] = check_arg_count(args)?;
                let left = this.read_immediate(left)?;
                let right = this.read_immediate(right)?;
                this.binop_ignore_overflow(mir::BinOp::Ne, &left, &right, dest)?;
            }
            "const_allocate" => {
                // For now, for compatibility with the run-time implementation of this, we just return null.
                // See <https://github.com/rust-lang/rust/issues/93935>.
                this.write_null(dest)?;
            }
            "const_deallocate" => {
                // complete NOP
            }

            // Raw memory accesses
            "volatile_load" => {
                let &[ref place] = check_arg_count(args)?;
                let place = this.deref_operand(place)?;
                this.copy_op(&place.into(), dest)?;
            }
            "volatile_store" => {
                let &[ref place, ref dest] = check_arg_count(args)?;
                let place = this.deref_operand(place)?;
                this.copy_op(dest, &place.into())?;
            }

            "write_bytes" | "volatile_set_memory" => {
                let &[ref ptr, ref val_byte, ref count] = check_arg_count(args)?;
                let ty = instance.substs.type_at(0);
                let ty_layout = this.layout_of(ty)?;
                let val_byte = this.read_scalar(val_byte)?.to_u8()?;
                let ptr = this.read_pointer(ptr)?;
                let count = this.read_scalar(count)?.to_machine_usize(this)?;
                let byte_count = ty_layout.size.checked_mul(count, this).ok_or_else(|| {
                    err_ub_format!("overflow computing total size of `{}`", intrinsic_name)
                })?;
                this.memory
                    .write_bytes(ptr, iter::repeat(val_byte).take(byte_count.bytes() as usize))?;
            }

            // Floating-point operations
            "fabsf32" => {
                let &[ref f] = check_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f32()?;
                // Can be implemented in soft-floats.
                this.write_scalar(Scalar::from_f32(f.abs()), dest)?;
            }
            "fabsf64" => {
                let &[ref f] = check_arg_count(args)?;
                let f = this.read_scalar(f)?.to_f64()?;
                // Can be implemented in soft-floats.
                this.write_scalar(Scalar::from_f64(f.abs()), dest)?;
            }
            #[rustfmt::skip]
            | "sinf32"
            | "cosf32"
            | "sqrtf32"
            | "expf32"
            | "exp2f32"
            | "logf32"
            | "log10f32"
            | "log2f32"
            | "floorf32"
            | "ceilf32"
            | "truncf32"
            | "roundf32"
            => {
                let &[ref f] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(f)?.to_u32()?);
                let f = match intrinsic_name {
                    "sinf32" => f.sin(),
                    "cosf32" => f.cos(),
                    "sqrtf32" => f.sqrt(),
                    "expf32" => f.exp(),
                    "exp2f32" => f.exp2(),
                    "logf32" => f.ln(),
                    "log10f32" => f.log10(),
                    "log2f32" => f.log2(),
                    "floorf32" => f.floor(),
                    "ceilf32" => f.ceil(),
                    "truncf32" => f.trunc(),
                    "roundf32" => f.round(),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u32(f.to_bits()), dest)?;
            }

            #[rustfmt::skip]
            | "sinf64"
            | "cosf64"
            | "sqrtf64"
            | "expf64"
            | "exp2f64"
            | "logf64"
            | "log10f64"
            | "log2f64"
            | "floorf64"
            | "ceilf64"
            | "truncf64"
            | "roundf64"
            => {
                let &[ref f] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(f)?.to_u64()?);
                let f = match intrinsic_name {
                    "sinf64" => f.sin(),
                    "cosf64" => f.cos(),
                    "sqrtf64" => f.sqrt(),
                    "expf64" => f.exp(),
                    "exp2f64" => f.exp2(),
                    "logf64" => f.ln(),
                    "log10f64" => f.log10(),
                    "log2f64" => f.log2(),
                    "floorf64" => f.floor(),
                    "ceilf64" => f.ceil(),
                    "truncf64" => f.trunc(),
                    "roundf64" => f.round(),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_u64(f.to_bits()), dest)?;
            }

            #[rustfmt::skip]
            | "fadd_fast"
            | "fsub_fast"
            | "fmul_fast"
            | "fdiv_fast"
            | "frem_fast"
            => {
                let &[ref a, ref b] = check_arg_count(args)?;
                let a = this.read_immediate(a)?;
                let b = this.read_immediate(b)?;
                let op = match intrinsic_name {
                    "fadd_fast" => mir::BinOp::Add,
                    "fsub_fast" => mir::BinOp::Sub,
                    "fmul_fast" => mir::BinOp::Mul,
                    "fdiv_fast" => mir::BinOp::Div,
                    "frem_fast" => mir::BinOp::Rem,
                    _ => bug!(),
                };
                let float_finite = |x: ImmTy<'tcx, _>| -> InterpResult<'tcx, bool> {
                    Ok(match x.layout.ty.kind() {
                        ty::Float(FloatTy::F32) => x.to_scalar()?.to_f32()?.is_finite(),
                        ty::Float(FloatTy::F64) => x.to_scalar()?.to_f64()?.is_finite(),
                        _ => bug!(
                            "`{}` called with non-float input type {:?}",
                            intrinsic_name,
                            x.layout.ty
                        ),
                    })
                };
                match (float_finite(a)?, float_finite(b)?) {
                    (false, false) => throw_ub_format!(
                        "`{}` intrinsic called with non-finite value as both parameters",
                        intrinsic_name,
                    ),
                    (false, _) => throw_ub_format!(
                        "`{}` intrinsic called with non-finite value as first parameter",
                        intrinsic_name,
                    ),
                    (_, false) => throw_ub_format!(
                        "`{}` intrinsic called with non-finite value as second parameter",
                        intrinsic_name,
                    ),
                    _ => {}
                }
                this.binop_ignore_overflow(op, &a, &b, dest)?;
            }

            #[rustfmt::skip]
            | "minnumf32"
            | "maxnumf32"
            | "copysignf32"
            => {
                let &[ref a, ref b] = check_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f32()?;
                let b = this.read_scalar(b)?.to_f32()?;
                let res = match intrinsic_name {
                    "minnumf32" => a.min(b),
                    "maxnumf32" => a.max(b),
                    "copysignf32" => a.copy_sign(b),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_f32(res), dest)?;
            }

            #[rustfmt::skip]
            | "minnumf64"
            | "maxnumf64"
            | "copysignf64"
            => {
                let &[ref a, ref b] = check_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f64()?;
                let b = this.read_scalar(b)?.to_f64()?;
                let res = match intrinsic_name {
                    "minnumf64" => a.min(b),
                    "maxnumf64" => a.max(b),
                    "copysignf64" => a.copy_sign(b),
                    _ => bug!(),
                };
                this.write_scalar(Scalar::from_f64(res), dest)?;
            }

            "powf32" => {
                let &[ref f, ref f2] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(f)?.to_u32()?);
                let f2 = f32::from_bits(this.read_scalar(f2)?.to_u32()?);
                this.write_scalar(Scalar::from_u32(f.powf(f2).to_bits()), dest)?;
            }

            "powf64" => {
                let &[ref f, ref f2] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(f)?.to_u64()?);
                let f2 = f64::from_bits(this.read_scalar(f2)?.to_u64()?);
                this.write_scalar(Scalar::from_u64(f.powf(f2).to_bits()), dest)?;
            }

            "fmaf32" => {
                let &[ref a, ref b, ref c] = check_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f32()?;
                let b = this.read_scalar(b)?.to_f32()?;
                let c = this.read_scalar(c)?.to_f32()?;
                let res = a.mul_add(b, c).value;
                this.write_scalar(Scalar::from_f32(res), dest)?;
            }

            "fmaf64" => {
                let &[ref a, ref b, ref c] = check_arg_count(args)?;
                let a = this.read_scalar(a)?.to_f64()?;
                let b = this.read_scalar(b)?.to_f64()?;
                let c = this.read_scalar(c)?.to_f64()?;
                let res = a.mul_add(b, c).value;
                this.write_scalar(Scalar::from_f64(res), dest)?;
            }

            "powif32" => {
                let &[ref f, ref i] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f32::from_bits(this.read_scalar(f)?.to_u32()?);
                let i = this.read_scalar(i)?.to_i32()?;
                this.write_scalar(Scalar::from_u32(f.powi(i).to_bits()), dest)?;
            }

            "powif64" => {
                let &[ref f, ref i] = check_arg_count(args)?;
                // FIXME: Using host floats.
                let f = f64::from_bits(this.read_scalar(f)?.to_u64()?);
                let i = this.read_scalar(i)?.to_i32()?;
                this.write_scalar(Scalar::from_u64(f.powi(i).to_bits()), dest)?;
            }

            "float_to_int_unchecked" => {
                let &[ref val] = check_arg_count(args)?;
                let val = this.read_immediate(val)?;

                let res = match val.layout.ty.kind() {
                    ty::Float(FloatTy::F32) =>
                        this.float_to_int_unchecked(val.to_scalar()?.to_f32()?, dest.layout.ty)?,
                    ty::Float(FloatTy::F64) =>
                        this.float_to_int_unchecked(val.to_scalar()?.to_f64()?, dest.layout.ty)?,
                    _ =>
                        bug!(
                            "`float_to_int_unchecked` called with non-float input type {:?}",
                            val.layout.ty
                        ),
                };

                this.write_scalar(res, dest)?;
            }

            // SIMD operations
            #[rustfmt::skip]
            | "simd_neg"
            | "simd_fabs"
            | "simd_ceil"
            | "simd_floor"
            | "simd_round"
            | "simd_trunc"
            | "simd_fsqrt" => {
                let &[ref op] = check_arg_count(args)?;
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
                    "simd_neg" => Op::MirOp(mir::UnOp::Neg),
                    "simd_fabs" => Op::Abs,
                    "simd_ceil" => Op::HostOp(HostFloatOp::Ceil),
                    "simd_floor" => Op::HostOp(HostFloatOp::Floor),
                    "simd_round" => Op::HostOp(HostFloatOp::Round),
                    "simd_trunc" => Op::HostOp(HostFloatOp::Trunc),
                    "simd_fsqrt" => Op::HostOp(HostFloatOp::Sqrt),
                    _ => unreachable!(),
                };

                for i in 0..dest_len {
                    let op = this.read_immediate(&this.mplace_index(&op, i)?.into())?;
                    let dest = this.mplace_index(&dest, i)?;
                    let val = match which {
                        Op::MirOp(mir_op) => this.unary_op(mir_op, &op)?.to_scalar()?,
                        Op::Abs => {
                            // Works for f32 and f64.
                            let ty::Float(float_ty) = op.layout.ty.kind() else {
                                bug!("{} operand is not a float", intrinsic_name)
                            };
                            let op = op.to_scalar()?;
                            match float_ty {
                                FloatTy::F32 => Scalar::from_f32(op.to_f32()?.abs()),
                                FloatTy::F64 => Scalar::from_f64(op.to_f64()?.abs()),
                            }
                        }
                        Op::HostOp(host_op) => {
                            let ty::Float(float_ty) = op.layout.ty.kind() else {
                                bug!("{} operand is not a float", intrinsic_name)
                            };
                            // FIXME using host floats
                            match float_ty {
                                FloatTy::F32 => {
                                    let f = f32::from_bits(op.to_scalar()?.to_u32()?);
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
                                    let f = f64::from_bits(op.to_scalar()?.to_u64()?);
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
            | "simd_add"
            | "simd_sub"
            | "simd_mul"
            | "simd_div"
            | "simd_rem"
            | "simd_shl"
            | "simd_shr"
            | "simd_and"
            | "simd_or"
            | "simd_xor"
            | "simd_eq"
            | "simd_ne"
            | "simd_lt"
            | "simd_le"
            | "simd_gt"
            | "simd_ge"
            | "simd_fmax"
            | "simd_fmin"
            | "simd_saturating_add"
            | "simd_saturating_sub" => {
                use mir::BinOp;

                let &[ref left, ref right] = check_arg_count(args)?;
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
                }
                let which = match intrinsic_name {
                    "simd_add" => Op::MirOp(BinOp::Add),
                    "simd_sub" => Op::MirOp(BinOp::Sub),
                    "simd_mul" => Op::MirOp(BinOp::Mul),
                    "simd_div" => Op::MirOp(BinOp::Div),
                    "simd_rem" => Op::MirOp(BinOp::Rem),
                    "simd_shl" => Op::MirOp(BinOp::Shl),
                    "simd_shr" => Op::MirOp(BinOp::Shr),
                    "simd_and" => Op::MirOp(BinOp::BitAnd),
                    "simd_or" => Op::MirOp(BinOp::BitOr),
                    "simd_xor" => Op::MirOp(BinOp::BitXor),
                    "simd_eq" => Op::MirOp(BinOp::Eq),
                    "simd_ne" => Op::MirOp(BinOp::Ne),
                    "simd_lt" => Op::MirOp(BinOp::Lt),
                    "simd_le" => Op::MirOp(BinOp::Le),
                    "simd_gt" => Op::MirOp(BinOp::Gt),
                    "simd_ge" => Op::MirOp(BinOp::Ge),
                    "simd_fmax" => Op::FMax,
                    "simd_fmin" => Op::FMin,
                    "simd_saturating_add" => Op::SaturatingOp(BinOp::Add),
                    "simd_saturating_sub" => Op::SaturatingOp(BinOp::Sub),
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
                                    let r_val = right.to_scalar()?.to_bits(right.layout.size)?;
                                    throw_ub_format!("overflowing shift by {} in `{}` in SIMD lane {}", r_val, intrinsic_name, i);
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
                        Op::FMax => {
                            fmax_op(&left, &right)?
                        }
                        Op::FMin => {
                            fmin_op(&left, &right)?
                        }
                        Op::SaturatingOp(mir_op) => {
                            this.saturating_arith(mir_op, &left, &right)?
                        }
                    };
                    this.write_scalar(val, &dest.into())?;
                }
            }
            "simd_fma" => {
                let &[ref a, ref b, ref c] = check_arg_count(args)?;
                let (a, a_len) = this.operand_to_simd(a)?;
                let (b, b_len) = this.operand_to_simd(b)?;
                let (c, c_len) = this.operand_to_simd(c)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, a_len);
                assert_eq!(dest_len, b_len);
                assert_eq!(dest_len, c_len);

                for i in 0..dest_len {
                    let a = this.read_immediate(&this.mplace_index(&a, i)?.into())?.to_scalar()?;
                    let b = this.read_immediate(&this.mplace_index(&b, i)?.into())?.to_scalar()?;
                    let c = this.read_immediate(&this.mplace_index(&c, i)?.into())?.to_scalar()?;
                    let dest = this.mplace_index(&dest, i)?;

                    // Works for f32 and f64.
                    let ty::Float(float_ty) = dest.layout.ty.kind() else {
                        bug!("{} operand is not a float", intrinsic_name)
                    };
                    let val = match float_ty {
                        FloatTy::F32 =>
                            Scalar::from_f32(a.to_f32()?.mul_add(b.to_f32()?, c.to_f32()?).value),
                        FloatTy::F64 =>
                            Scalar::from_f64(a.to_f64()?.mul_add(b.to_f64()?, c.to_f64()?).value),
                    };
                    this.write_scalar(val, &dest.into())?;
                }
            }
            #[rustfmt::skip]
            | "simd_reduce_and"
            | "simd_reduce_or"
            | "simd_reduce_xor"
            | "simd_reduce_any"
            | "simd_reduce_all"
            | "simd_reduce_max"
            | "simd_reduce_min" => {
                use mir::BinOp;

                let &[ref op] = check_arg_count(args)?;
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
                    "simd_reduce_and" => Op::MirOp(BinOp::BitAnd),
                    "simd_reduce_or" => Op::MirOp(BinOp::BitOr),
                    "simd_reduce_xor" => Op::MirOp(BinOp::BitXor),
                    "simd_reduce_any" => Op::MirOpBool(BinOp::BitOr),
                    "simd_reduce_all" => Op::MirOpBool(BinOp::BitAnd),
                    "simd_reduce_max" => Op::Max,
                    "simd_reduce_min" => Op::Min,
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
                                if this.binary_op(BinOp::Ge, &res, &op)?.to_scalar()?.to_bool()? {
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
                                if this.binary_op(BinOp::Le, &res, &op)?.to_scalar()?.to_bool()? {
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
            | "simd_reduce_add_ordered"
            | "simd_reduce_mul_ordered" => {
                use mir::BinOp;

                let &[ref op, ref init] = check_arg_count(args)?;
                let (op, op_len) = this.operand_to_simd(op)?;
                let init = this.read_immediate(init)?;

                let mir_op = match intrinsic_name {
                    "simd_reduce_add_ordered" => BinOp::Add,
                    "simd_reduce_mul_ordered" => BinOp::Mul,
                    _ => unreachable!(),
                };

                let mut res = init;
                for i in 0..op_len {
                    let op = this.read_immediate(&this.mplace_index(&op, i)?.into())?;
                    res = this.binary_op(mir_op, &res, &op)?;
                }
                this.write_immediate(*res, dest)?;
            }
            "simd_select" => {
                let &[ref mask, ref yes, ref no] = check_arg_count(args)?;
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
            "simd_select_bitmask" => {
                let &[ref mask, ref yes, ref no] = check_arg_count(args)?;
                let (yes, yes_len) = this.operand_to_simd(yes)?;
                let (no, no_len) = this.operand_to_simd(no)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;
                let bitmask_len = dest_len.max(8);

                assert!(mask.layout.ty.is_integral());
                assert!(bitmask_len <= 64);
                assert_eq!(bitmask_len, mask.layout.size.bits());
                assert_eq!(dest_len, yes_len);
                assert_eq!(dest_len, no_len);

                let mask: u64 = this
                    .read_scalar(mask)?
                    .check_init()?
                    .to_bits(mask.layout.size)?
                    .try_into()
                    .unwrap();
                for i in 0..dest_len {
                    let mask =
                        mask & (1 << simd_bitmask_index(i, dest_len, this.data_layout().endian));
                    let yes = this.read_immediate(&this.mplace_index(&yes, i)?.into())?;
                    let no = this.read_immediate(&this.mplace_index(&no, i)?.into())?;
                    let dest = this.mplace_index(&dest, i)?;

                    let val = if mask != 0 { yes } else { no };
                    this.write_immediate(*val, &dest.into())?;
                }
                for i in dest_len..bitmask_len {
                    // If the mask is "padded", ensure that padding is all-zero.
                    let mask = mask & (1 << i);
                    if mask != 0 {
                        throw_ub_format!(
                            "a SIMD bitmask less than 8 bits long must be filled with 0s for the remaining bits"
                        );
                    }
                }
            }
            #[rustfmt::skip]
            "simd_cast" | "simd_as" => {
                let &[ref op] = check_arg_count(args)?;
                let (op, op_len) = this.operand_to_simd(op)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                assert_eq!(dest_len, op_len);

                let safe_cast = intrinsic_name == "simd_as";

                for i in 0..dest_len {
                    let op = this.read_immediate(&this.mplace_index(&op, i)?.into())?;
                    let dest = this.mplace_index(&dest, i)?;

                    let val = match (op.layout.ty.kind(), dest.layout.ty.kind()) {
                        // Int-to-(int|float): always safe
                        (ty::Int(_) | ty::Uint(_), ty::Int(_) | ty::Uint(_) | ty::Float(_)) =>
                            this.misc_cast(&op, dest.layout.ty)?,
                        // Float-to-float: always safe
                        (ty::Float(_), ty::Float(_)) =>
                            this.misc_cast(&op, dest.layout.ty)?,
                        // Float-to-int in safe mode
                        (ty::Float(_), ty::Int(_) | ty::Uint(_)) if safe_cast =>
                            this.misc_cast(&op, dest.layout.ty)?,
                        // Float-to-int in unchecked mode
                        (ty::Float(FloatTy::F32), ty::Int(_) | ty::Uint(_)) if !safe_cast =>
                            this.float_to_int_unchecked(op.to_scalar()?.to_f32()?, dest.layout.ty)?.into(),
                        (ty::Float(FloatTy::F64), ty::Int(_) | ty::Uint(_)) if !safe_cast =>
                            this.float_to_int_unchecked(op.to_scalar()?.to_f64()?, dest.layout.ty)?.into(),
                        _ =>
                            throw_unsup_format!(
                                "Unsupported SIMD cast from element type {} to {}",
                                op.layout.ty,
                                dest.layout.ty
                            ),
                    };
                    this.write_immediate(val, &dest.into())?;
                }
            }
            "simd_shuffle" => {
                let &[ref left, ref right, ref index] = check_arg_count(args)?;
                let (left, left_len) = this.operand_to_simd(left)?;
                let (right, right_len) = this.operand_to_simd(right)?;
                let (dest, dest_len) = this.place_to_simd(dest)?;

                // `index` is an array, not a SIMD type
                let ty::Array(_, index_len) = index.layout.ty.kind() else {
                    bug!("simd_shuffle index argument has non-array type {}", index.layout.ty)
                };
                let index_len = index_len.eval_usize(*this.tcx, this.param_env());

                assert_eq!(left_len, right_len);
                assert_eq!(index_len, dest_len);

                for i in 0..dest_len {
                    let src_index: u64 = this
                        .read_immediate(&this.operand_index(&index, i)?.into())?
                        .to_scalar()?
                        .to_u32()?
                        .into();
                    let dest = this.mplace_index(&dest, i)?;

                    let val = if src_index < left_len {
                        this.read_immediate(&this.mplace_index(&left, src_index)?.into())?
                    } else if src_index < left_len.checked_add(right_len).unwrap() {
                        this.read_immediate(
                            &this.mplace_index(&right, src_index - left_len)?.into(),
                        )?
                    } else {
                        bug!(
                            "simd_shuffle index {} is out of bounds for 2 vectors of size {}",
                            src_index,
                            left_len
                        );
                    };
                    this.write_immediate(*val, &dest.into())?;
                }
            }
            "simd_gather" => {
                let &[ref passthru, ref ptrs, ref mask] = check_arg_count(args)?;
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
            "simd_scatter" => {
                let &[ref value, ref ptrs, ref mask] = check_arg_count(args)?;
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
            "simd_bitmask" => {
                let &[ref op] = check_arg_count(args)?;
                let (op, op_len) = this.operand_to_simd(op)?;
                let bitmask_len = op_len.max(8);

                assert!(dest.layout.ty.is_integral());
                assert!(bitmask_len <= 64);
                assert_eq!(bitmask_len, dest.layout.size.bits());

                let mut res = 0u64;
                for i in 0..op_len {
                    let op = this.read_immediate(&this.mplace_index(&op, i)?.into())?;
                    if simd_element_to_bool(op)? {
                        res |= 1 << simd_bitmask_index(i, op_len, this.data_layout().endian);
                    }
                }
                this.write_int(res, dest)?;
            }

            // Atomic operations
            "atomic_load" => this.atomic_load(args, dest, AtomicReadOp::SeqCst)?,
            "atomic_load_relaxed" => this.atomic_load(args, dest, AtomicReadOp::Relaxed)?,
            "atomic_load_acq" => this.atomic_load(args, dest, AtomicReadOp::Acquire)?,

            "atomic_store" => this.atomic_store(args, AtomicWriteOp::SeqCst)?,
            "atomic_store_relaxed" => this.atomic_store(args, AtomicWriteOp::Relaxed)?,
            "atomic_store_rel" => this.atomic_store(args, AtomicWriteOp::Release)?,

            "atomic_fence_acq" => this.atomic_fence(args, AtomicFenceOp::Acquire)?,
            "atomic_fence_rel" => this.atomic_fence(args, AtomicFenceOp::Release)?,
            "atomic_fence_acqrel" => this.atomic_fence(args, AtomicFenceOp::AcqRel)?,
            "atomic_fence" => this.atomic_fence(args, AtomicFenceOp::SeqCst)?,

            "atomic_singlethreadfence_acq" => this.compiler_fence(args, AtomicFenceOp::Acquire)?,
            "atomic_singlethreadfence_rel" => this.compiler_fence(args, AtomicFenceOp::Release)?,
            "atomic_singlethreadfence_acqrel" =>
                this.compiler_fence(args, AtomicFenceOp::AcqRel)?,
            "atomic_singlethreadfence" => this.compiler_fence(args, AtomicFenceOp::SeqCst)?,

            "atomic_xchg" => this.atomic_exchange(args, dest, AtomicRwOp::SeqCst)?,
            "atomic_xchg_acq" => this.atomic_exchange(args, dest, AtomicRwOp::Acquire)?,
            "atomic_xchg_rel" => this.atomic_exchange(args, dest, AtomicRwOp::Release)?,
            "atomic_xchg_acqrel" => this.atomic_exchange(args, dest, AtomicRwOp::AcqRel)?,
            "atomic_xchg_relaxed" => this.atomic_exchange(args, dest, AtomicRwOp::Relaxed)?,

            #[rustfmt::skip]
            "atomic_cxchg" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::SeqCst, AtomicReadOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_cxchg_acq" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::Acquire, AtomicReadOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_cxchg_rel" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::Release, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchg_acqrel" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::AcqRel, AtomicReadOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_cxchg_relaxed" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::Relaxed, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchg_acq_failrelaxed" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::Acquire, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchg_acqrel_failrelaxed" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::AcqRel, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchg_failrelaxed" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::SeqCst, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchg_failacq" =>
                this.atomic_compare_exchange(args, dest, AtomicRwOp::SeqCst, AtomicReadOp::Acquire)?,

            #[rustfmt::skip]
            "atomic_cxchgweak" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::SeqCst, AtomicReadOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_acq" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::Acquire, AtomicReadOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_rel" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::Release, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_acqrel" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::AcqRel, AtomicReadOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_relaxed" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::Relaxed, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_acq_failrelaxed" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::Acquire, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_acqrel_failrelaxed" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::AcqRel, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_failrelaxed" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::SeqCst, AtomicReadOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_cxchgweak_failacq" =>
                this.atomic_compare_exchange_weak(args, dest, AtomicRwOp::SeqCst, AtomicReadOp::Acquire)?,

            #[rustfmt::skip]
            "atomic_or" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_or_acq" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_or_rel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOp::Release)?,
            #[rustfmt::skip]
            "atomic_or_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOp::AcqRel)?,
            #[rustfmt::skip]
            "atomic_or_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitOr, false), AtomicRwOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_xor" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_xor_acq" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_xor_rel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOp::Release)?,
            #[rustfmt::skip]
            "atomic_xor_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOp::AcqRel)?,
            #[rustfmt::skip]
            "atomic_xor_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitXor, false), AtomicRwOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_and" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_and_acq" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_and_rel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOp::Release)?,
            #[rustfmt::skip]
            "atomic_and_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOp::AcqRel)?,
            #[rustfmt::skip]
            "atomic_and_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, false), AtomicRwOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_nand" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_nand_acq" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_nand_rel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOp::Release)?,
            #[rustfmt::skip]
            "atomic_nand_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOp::AcqRel)?,
            #[rustfmt::skip]
            "atomic_nand_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::BitAnd, true), AtomicRwOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_xadd" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_xadd_acq" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_xadd_rel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOp::Release)?,
            #[rustfmt::skip]
            "atomic_xadd_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOp::AcqRel)?,
            #[rustfmt::skip]
            "atomic_xadd_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Add, false), AtomicRwOp::Relaxed)?,
            #[rustfmt::skip]
            "atomic_xsub" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOp::SeqCst)?,
            #[rustfmt::skip]
            "atomic_xsub_acq" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOp::Acquire)?,
            #[rustfmt::skip]
            "atomic_xsub_rel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOp::Release)?,
            #[rustfmt::skip]
            "atomic_xsub_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOp::AcqRel)?,
            #[rustfmt::skip]
            "atomic_xsub_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::MirOp(BinOp::Sub, false), AtomicRwOp::Relaxed)?,
            "atomic_min" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::SeqCst)?,
            "atomic_min_acq" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::Acquire)?,
            "atomic_min_rel" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::Release)?,
            "atomic_min_acqrel" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::AcqRel)?,
            "atomic_min_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::Relaxed)?,
            "atomic_max" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::SeqCst)?,
            "atomic_max_acq" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::Acquire)?,
            "atomic_max_rel" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::Release)?,
            "atomic_max_acqrel" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::AcqRel)?,
            "atomic_max_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::Relaxed)?,
            "atomic_umin" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::SeqCst)?,
            "atomic_umin_acq" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::Acquire)?,
            "atomic_umin_rel" => this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::Release)?,
            "atomic_umin_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::AcqRel)?,
            "atomic_umin_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::Min, AtomicRwOp::Relaxed)?,
            "atomic_umax" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::SeqCst)?,
            "atomic_umax_acq" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::Acquire)?,
            "atomic_umax_rel" => this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::Release)?,
            "atomic_umax_acqrel" =>
                this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::AcqRel)?,
            "atomic_umax_relaxed" =>
                this.atomic_op(args, dest, AtomicOp::Max, AtomicRwOp::Relaxed)?,

            // Other
            "exact_div" => {
                let &[ref num, ref denom] = check_arg_count(args)?;
                this.exact_div(&this.read_immediate(num)?, &this.read_immediate(denom)?, dest)?;
            }

            "try" => return this.handle_try(args, dest, ret),

            "breakpoint" => {
                let &[] = check_arg_count(args)?;
                // normally this would raise a SIGTRAP, which aborts if no debugger is connected
                throw_machine_stop!(TerminationInfo::Abort("Trace/breakpoint trap".to_string()))
            }

            name => throw_unsup_format!("unimplemented intrinsic: {}", name),
        }

        trace!("{:?}", this.dump_place(**dest));
        this.go_to_block(ret);
        Ok(())
    }

    fn atomic_load(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        atomic: AtomicReadOp,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let &[ref place] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;

        // make sure it fits into a scalar; otherwise it cannot be atomic
        let val = this.read_scalar_atomic(&place, atomic)?;

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.memory.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;
        // Perform regular access.
        this.write_scalar(val, dest)?;
        Ok(())
    }

    fn atomic_store(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        atomic: AtomicWriteOp,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let &[ref place, ref val] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;
        let val = this.read_scalar(val)?; // make sure it fits into a scalar; otherwise it cannot be atomic

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.memory.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;

        // Perform atomic store
        this.write_scalar_atomic(val, &place, atomic)?;
        Ok(())
    }

    fn compiler_fence(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        atomic: AtomicFenceOp,
    ) -> InterpResult<'tcx> {
        let &[] = check_arg_count(args)?;
        let _ = atomic;
        //FIXME: compiler fences are currently ignored
        Ok(())
    }

    fn atomic_fence(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        atomic: AtomicFenceOp,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let &[] = check_arg_count(args)?;
        this.validate_atomic_fence(atomic)?;
        Ok(())
    }

    fn atomic_op(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        atomic_op: AtomicOp,
        atomic: AtomicRwOp,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let &[ref place, ref rhs] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;

        if !place.layout.ty.is_integral() {
            bug!("Atomic arithmetic operations only work on integer types");
        }
        let rhs = this.read_immediate(rhs)?;

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.memory.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;

        match atomic_op {
            AtomicOp::Min => {
                let old = this.atomic_min_max_scalar(&place, rhs, true, atomic)?;
                this.write_immediate(*old, &dest)?; // old value is returned
                Ok(())
            }
            AtomicOp::Max => {
                let old = this.atomic_min_max_scalar(&place, rhs, false, atomic)?;
                this.write_immediate(*old, &dest)?; // old value is returned
                Ok(())
            }
            AtomicOp::MirOp(op, neg) => {
                let old = this.atomic_op_immediate(&place, &rhs, op, neg, atomic)?;
                this.write_immediate(*old, dest)?; // old value is returned
                Ok(())
            }
        }
    }

    fn atomic_exchange(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        atomic: AtomicRwOp,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let &[ref place, ref new] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;
        let new = this.read_scalar(new)?;

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.memory.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;

        let old = this.atomic_exchange_scalar(&place, new, atomic)?;
        this.write_scalar(old, dest)?; // old value is returned
        Ok(())
    }

    fn atomic_compare_exchange_impl(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        success: AtomicRwOp,
        fail: AtomicReadOp,
        can_fail_spuriously: bool,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let &[ref place, ref expect_old, ref new] = check_arg_count(args)?;
        let place = this.deref_operand(place)?;
        let expect_old = this.read_immediate(expect_old)?; // read as immediate for the sake of `binary_op()`
        let new = this.read_scalar(new)?;

        // Check alignment requirements. Atomics must always be aligned to their size,
        // even if the type they wrap would be less aligned (e.g. AtomicU64 on 32bit must
        // be 8-aligned).
        let align = Align::from_bytes(place.layout.size.bytes()).unwrap();
        this.memory.check_ptr_access_align(
            place.ptr,
            place.layout.size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
        )?;

        let old = this.atomic_compare_exchange_scalar(
            &place,
            &expect_old,
            new,
            success,
            fail,
            can_fail_spuriously,
        )?;

        // Return old value.
        this.write_immediate(old, dest)?;
        Ok(())
    }

    fn atomic_compare_exchange(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        success: AtomicRwOp,
        fail: AtomicReadOp,
    ) -> InterpResult<'tcx> {
        self.atomic_compare_exchange_impl(args, dest, success, fail, false)
    }

    fn atomic_compare_exchange_weak(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        success: AtomicRwOp,
        fail: AtomicReadOp,
    ) -> InterpResult<'tcx> {
        self.atomic_compare_exchange_impl(args, dest, success, fail, true)
    }

    fn float_to_int_unchecked<F>(
        &self,
        f: F,
        dest_ty: ty::Ty<'tcx>,
    ) -> InterpResult<'tcx, Scalar<Tag>>
    where
        F: Float + Into<Scalar<Tag>>,
    {
        let this = self.eval_context_ref();

        // Step 1: cut off the fractional part of `f`. The result of this is
        // guaranteed to be precisely representable in IEEE floats.
        let f = f.round_to_integral(Round::TowardZero).value;

        // Step 2: Cast the truncated float to the target integer type and see if we lose any information in this step.
        Ok(match dest_ty.kind() {
            // Unsigned
            ty::Uint(t) => {
                let size = Integer::from_uint_ty(this, *t).size();
                let res = f.to_u128(size.bits_usize());
                if res.status.is_empty() {
                    // No status flags means there was no further rounding or other loss of precision.
                    Scalar::from_uint(res.value, size)
                } else {
                    // `f` was not representable in this integer type.
                    throw_ub_format!(
                        "`float_to_int_unchecked` intrinsic called on {} which cannot be represented in target type `{:?}`",
                        f,
                        dest_ty,
                    );
                }
            }
            // Signed
            ty::Int(t) => {
                let size = Integer::from_int_ty(this, *t).size();
                let res = f.to_i128(size.bits_usize());
                if res.status.is_empty() {
                    // No status flags means there was no further rounding or other loss of precision.
                    Scalar::from_int(res.value, size)
                } else {
                    // `f` was not representable in this integer type.
                    throw_ub_format!(
                        "`float_to_int_unchecked` intrinsic called on {} which cannot be represented in target type `{:?}`",
                        f,
                        dest_ty,
                    );
                }
            }
            // Nothing else
            _ => bug!("`float_to_int_unchecked` called with non-int output type {:?}", dest_ty),
        })
    }
}

fn fmax_op<'tcx>(
    left: &ImmTy<'tcx, Tag>,
    right: &ImmTy<'tcx, Tag>,
) -> InterpResult<'tcx, Scalar<Tag>> {
    assert_eq!(left.layout.ty, right.layout.ty);
    let ty::Float(float_ty) = left.layout.ty.kind() else {
        bug!("fmax operand is not a float")
    };
    let left = left.to_scalar()?;
    let right = right.to_scalar()?;
    Ok(match float_ty {
        FloatTy::F32 => Scalar::from_f32(left.to_f32()?.max(right.to_f32()?)),
        FloatTy::F64 => Scalar::from_f64(left.to_f64()?.max(right.to_f64()?)),
    })
}

fn fmin_op<'tcx>(
    left: &ImmTy<'tcx, Tag>,
    right: &ImmTy<'tcx, Tag>,
) -> InterpResult<'tcx, Scalar<Tag>> {
    assert_eq!(left.layout.ty, right.layout.ty);
    let ty::Float(float_ty) = left.layout.ty.kind() else {
        bug!("fmin operand is not a float")
    };
    let left = left.to_scalar()?;
    let right = right.to_scalar()?;
    Ok(match float_ty {
        FloatTy::F32 => Scalar::from_f32(left.to_f32()?.min(right.to_f32()?)),
        FloatTy::F64 => Scalar::from_f64(left.to_f64()?.min(right.to_f64()?)),
    })
}

fn bool_to_simd_element(b: bool, size: Size) -> Scalar<Tag> {
    // SIMD uses all-1 as pattern for "true"
    let val = if b { -1 } else { 0 };
    Scalar::from_int(val, size)
}

fn simd_element_to_bool<'tcx>(elem: ImmTy<'tcx, Tag>) -> InterpResult<'tcx, bool> {
    let val = elem.to_scalar()?.to_int(elem.layout.size)?;
    Ok(match val {
        0 => false,
        -1 => true,
        _ => throw_ub_format!("each element of a SIMD mask must be all-0-bits or all-1-bits"),
    })
}

fn simd_bitmask_index(idx: u64, vec_len: u64, endianess: Endian) -> u64 {
    assert!(idx < vec_len);
    match endianess {
        Endian::Little => idx,
        Endian::Big => vec_len - 1 - idx, // reverse order of bits
    }
}
