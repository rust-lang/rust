//! Codegen SIMD intrinsics.

use cranelift_codegen::ir::immediates::Offset32;
use rustc_abi::Endian;

use super::*;
use crate::prelude::*;

fn report_simd_type_validation_error(
    fx: &mut FunctionCx<'_, '_, '_>,
    intrinsic: Symbol,
    span: Span,
    ty: Ty<'_>,
) {
    fx.tcx.dcx().span_err(span, format!("invalid monomorphization of `{}` intrinsic: expected SIMD input type, found non-SIMD `{}`", intrinsic, ty));
    // Prevent verifier error
    fx.bcx.ins().trap(TrapCode::user(1 /* unreachable */).unwrap());
}

pub(super) fn codegen_simd_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    intrinsic: Symbol,
    generic_args: GenericArgsRef<'tcx>,
    args: &[Spanned<mir::Operand<'tcx>>],
    ret: CPlace<'tcx>,
    target: BasicBlock,
    span: Span,
) {
    match intrinsic {
        sym::simd_as | sym::simd_cast => {
            intrinsic_args!(fx, args => (a); intrinsic);

            if !a.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, a.layout().ty);
                return;
            }

            simd_for_each_lane(fx, a, ret, &|fx, lane_ty, ret_lane_ty, lane| {
                let ret_lane_clif_ty = fx.clif_type(ret_lane_ty).unwrap();

                let from_signed = type_sign(lane_ty);
                let to_signed = type_sign(ret_lane_ty);

                clif_int_or_float_cast(fx, lane, from_signed, ret_lane_clif_ty, to_signed)
            });
        }

        sym::simd_eq | sym::simd_ne | sym::simd_lt | sym::simd_le | sym::simd_gt | sym::simd_ge => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            if !x.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, x.layout().ty);
                return;
            }

            // FIXME use vector instructions when possible
            simd_pair_for_each_lane(fx, x, y, ret, &|fx, lane_ty, res_lane_ty, x_lane, y_lane| {
                let res_lane = match (lane_ty.kind(), intrinsic) {
                    (ty::Uint(_), sym::simd_eq) => fx.bcx.ins().icmp(IntCC::Equal, x_lane, y_lane),
                    (ty::Uint(_), sym::simd_ne) => {
                        fx.bcx.ins().icmp(IntCC::NotEqual, x_lane, y_lane)
                    }
                    (ty::Uint(_), sym::simd_lt) => {
                        fx.bcx.ins().icmp(IntCC::UnsignedLessThan, x_lane, y_lane)
                    }
                    (ty::Uint(_), sym::simd_le) => {
                        fx.bcx.ins().icmp(IntCC::UnsignedLessThanOrEqual, x_lane, y_lane)
                    }
                    (ty::Uint(_), sym::simd_gt) => {
                        fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan, x_lane, y_lane)
                    }
                    (ty::Uint(_), sym::simd_ge) => {
                        fx.bcx.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, x_lane, y_lane)
                    }

                    (ty::Int(_), sym::simd_eq) => fx.bcx.ins().icmp(IntCC::Equal, x_lane, y_lane),
                    (ty::Int(_), sym::simd_ne) => {
                        fx.bcx.ins().icmp(IntCC::NotEqual, x_lane, y_lane)
                    }
                    (ty::Int(_), sym::simd_lt) => {
                        fx.bcx.ins().icmp(IntCC::SignedLessThan, x_lane, y_lane)
                    }
                    (ty::Int(_), sym::simd_le) => {
                        fx.bcx.ins().icmp(IntCC::SignedLessThanOrEqual, x_lane, y_lane)
                    }
                    (ty::Int(_), sym::simd_gt) => {
                        fx.bcx.ins().icmp(IntCC::SignedGreaterThan, x_lane, y_lane)
                    }
                    (ty::Int(_), sym::simd_ge) => {
                        fx.bcx.ins().icmp(IntCC::SignedGreaterThanOrEqual, x_lane, y_lane)
                    }

                    (ty::Float(_), sym::simd_eq) => {
                        fx.bcx.ins().fcmp(FloatCC::Equal, x_lane, y_lane)
                    }
                    (ty::Float(_), sym::simd_ne) => {
                        fx.bcx.ins().fcmp(FloatCC::NotEqual, x_lane, y_lane)
                    }
                    (ty::Float(_), sym::simd_lt) => {
                        fx.bcx.ins().fcmp(FloatCC::LessThan, x_lane, y_lane)
                    }
                    (ty::Float(_), sym::simd_le) => {
                        fx.bcx.ins().fcmp(FloatCC::LessThanOrEqual, x_lane, y_lane)
                    }
                    (ty::Float(_), sym::simd_gt) => {
                        fx.bcx.ins().fcmp(FloatCC::GreaterThan, x_lane, y_lane)
                    }
                    (ty::Float(_), sym::simd_ge) => {
                        fx.bcx.ins().fcmp(FloatCC::GreaterThanOrEqual, x_lane, y_lane)
                    }

                    _ => unreachable!(),
                };

                bool_to_zero_or_max_uint(fx, res_lane_ty, res_lane)
            });
        }

        // simd_shuffle_const_generic<T, U, const I: &[u32]>(x: T, y: T) -> U
        sym::simd_shuffle_const_generic => {
            let [x, y] = args else {
                bug!("wrong number of args for intrinsic {intrinsic}");
            };
            let x = codegen_operand(fx, &x.node);
            let y = codegen_operand(fx, &y.node);

            if !x.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, x.layout().ty);
                return;
            }

            let idx = generic_args[2].expect_const().to_value().valtree.unwrap_branch();

            assert_eq!(x.layout(), y.layout());
            let layout = x.layout();

            let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);

            assert_eq!(lane_ty, ret_lane_ty);
            assert_eq!(idx.len() as u64, ret_lane_count);

            let total_len = lane_count * 2;

            let indexes = idx.iter().map(|idx| idx.unwrap_leaf().to_u32()).collect::<Vec<u32>>();

            for &idx in &indexes {
                assert!(u64::from(idx) < total_len, "idx {} out of range 0..{}", idx, total_len);
            }

            for (out_idx, in_idx) in indexes.into_iter().enumerate() {
                let in_lane = if u64::from(in_idx) < lane_count {
                    x.value_lane(fx, in_idx.into())
                } else {
                    y.value_lane(fx, u64::from(in_idx) - lane_count)
                };
                let out_lane = ret.place_lane(fx, u64::try_from(out_idx).unwrap());
                out_lane.write_cvalue(fx, in_lane);
            }
        }

        // simd_shuffle<T, I, U>(x: T, y: T, idx: I) -> U
        sym::simd_shuffle => {
            let (x, y, idx) = match args {
                [x, y, idx] => (x, y, idx),
                _ => {
                    bug!("wrong number of args for intrinsic {intrinsic}");
                }
            };
            let x = codegen_operand(fx, &x.node);
            let y = codegen_operand(fx, &y.node);

            if !x.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, x.layout().ty);
                return;
            }

            // Make sure this is actually a SIMD vector.
            let idx_ty = fx.monomorphize(idx.node.ty(fx.mir, fx.tcx));
            if !idx_ty.is_simd()
                || !matches!(idx_ty.simd_size_and_type(fx.tcx).1.kind(), ty::Uint(ty::UintTy::U32))
            {
                fx.tcx.dcx().span_err(
                    span,
                    format!("simd_shuffle index must be a SIMD vector of `u32`, got `{}`", idx_ty),
                );
                // Prevent verifier error
                fx.bcx.ins().trap(TrapCode::user(1 /* unreachable */).unwrap());
                return;
            };
            let n: u16 = idx_ty.simd_size_and_type(fx.tcx).0.try_into().unwrap();

            assert_eq!(x.layout(), y.layout());
            let layout = x.layout();

            let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);

            assert_eq!(lane_ty, ret_lane_ty);
            assert_eq!(u64::from(n), ret_lane_count);

            let total_len = lane_count * 2;

            // FIXME: this is a terrible abstraction-breaking hack.
            // Find a way to reuse `immediate_const_vector` from `codegen_ssa` instead.
            let indexes = {
                use rustc_middle::mir::interpret::*;
                let idx_const = if let Some(const_) = idx.node.constant() {
                    crate::constant::eval_mir_constant(fx, const_).0
                } else {
                    unreachable!("{idx:?}")
                };

                let idx_bytes = match idx_const {
                    ConstValue::Indirect { alloc_id, offset } => {
                        let alloc = fx.tcx.global_alloc(alloc_id).unwrap_memory();
                        let size = Size::from_bytes(
                            4 * ret_lane_count, /* size_of([u32; ret_lane_count]) */
                        );
                        alloc
                            .inner()
                            .get_bytes_strip_provenance(fx, alloc_range(offset, size))
                            .unwrap()
                    }
                    _ => unreachable!("{:?}", idx_const),
                };

                (0..ret_lane_count)
                    .map(|i| {
                        let i = usize::try_from(i).unwrap();
                        let idx = rustc_middle::mir::interpret::read_target_uint(
                            fx.tcx.data_layout.endian,
                            &idx_bytes[4 * i..4 * i + 4],
                        )
                        .expect("read_target_uint");
                        u16::try_from(idx).expect("try_from u32")
                    })
                    .collect::<Vec<u16>>()
            };

            for &idx in &indexes {
                assert!(u64::from(idx) < total_len, "idx {} out of range 0..{}", idx, total_len);
            }

            for (out_idx, in_idx) in indexes.into_iter().enumerate() {
                let in_lane = if u64::from(in_idx) < lane_count {
                    x.value_lane(fx, in_idx.into())
                } else {
                    y.value_lane(fx, u64::from(in_idx) - lane_count)
                };
                let out_lane = ret.place_lane(fx, u64::try_from(out_idx).unwrap());
                out_lane.write_cvalue(fx, in_lane);
            }
        }

        sym::simd_insert => {
            let (base, idx, val) = match args {
                [base, idx, val] => (base, idx, val),
                _ => {
                    bug!("wrong number of args for intrinsic {intrinsic}");
                }
            };
            let base = codegen_operand(fx, &base.node);
            let val = codegen_operand(fx, &val.node);

            // FIXME validate
            let idx_const = if let Some(idx_const) = idx.node.constant() {
                crate::constant::eval_mir_constant(fx, idx_const).0.try_to_scalar_int().unwrap()
            } else {
                fx.tcx.dcx().span_fatal(span, "Index argument for `simd_insert` is not a constant");
            };

            let idx: u32 = idx_const.to_u32();
            let (lane_count, _lane_ty) = base.layout().ty.simd_size_and_type(fx.tcx);
            if u64::from(idx) >= lane_count {
                fx.tcx.dcx().span_fatal(
                    fx.mir.span,
                    format!("[simd_insert] idx {} >= lane_count {}", idx, lane_count),
                );
            }

            ret.write_cvalue(fx, base);
            let ret_lane = ret.place_lane(fx, idx.into());
            ret_lane.write_cvalue(fx, val);
        }

        sym::simd_insert_dyn => {
            intrinsic_args!(fx, args => (base, idx, val); intrinsic);

            if !base.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, base.layout().ty);
                return;
            }

            let idx = idx.load_scalar(fx);

            ret.write_cvalue(fx, base);
            ret.write_lane_dyn(fx, idx, val);
        }

        sym::simd_extract => {
            let (v, idx) = match args {
                [v, idx] => (v, idx),
                _ => {
                    bug!("wrong number of args for intrinsic {intrinsic}");
                }
            };
            let v = codegen_operand(fx, &v.node);

            if !v.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, v.layout().ty);
                return;
            }

            let idx_const = if let Some(idx_const) = idx.node.constant() {
                crate::constant::eval_mir_constant(fx, idx_const).0.try_to_scalar_int().unwrap()
            } else {
                fx.tcx
                    .dcx()
                    .span_fatal(span, "Index argument for `simd_extract` is not a constant");
            };

            let idx = idx_const.to_u32();
            let (lane_count, _lane_ty) = v.layout().ty.simd_size_and_type(fx.tcx);
            if u64::from(idx) >= lane_count {
                fx.tcx.dcx().span_fatal(
                    fx.mir.span,
                    format!("[simd_extract] idx {} >= lane_count {}", idx, lane_count),
                );
            }

            let ret_lane = v.value_lane(fx, idx.into());
            ret.write_cvalue(fx, ret_lane);
        }

        sym::simd_extract_dyn => {
            intrinsic_args!(fx, args => (v, idx); intrinsic);

            if !v.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, v.layout().ty);
                return;
            }

            let idx = idx.load_scalar(fx);

            let ret_lane = v.value_lane_dyn(fx, idx);
            ret.write_cvalue(fx, ret_lane);
        }

        sym::simd_neg
        | sym::simd_bswap
        | sym::simd_bitreverse
        | sym::simd_ctlz
        | sym::simd_ctpop
        | sym::simd_cttz => {
            intrinsic_args!(fx, args => (a); intrinsic);

            if !a.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, a.layout().ty);
                return;
            }

            simd_for_each_lane(fx, a, ret, &|fx, lane_ty, _ret_lane_ty, lane| match (
                lane_ty.kind(),
                intrinsic,
            ) {
                (ty::Int(_), sym::simd_neg) => fx.bcx.ins().ineg(lane),
                (ty::Float(_), sym::simd_neg) => fx.bcx.ins().fneg(lane),

                (ty::Uint(ty::UintTy::U8) | ty::Int(ty::IntTy::I8), sym::simd_bswap) => lane,
                (ty::Uint(_) | ty::Int(_), sym::simd_bswap) => fx.bcx.ins().bswap(lane),
                (ty::Uint(_) | ty::Int(_), sym::simd_bitreverse) => fx.bcx.ins().bitrev(lane),
                (ty::Uint(_) | ty::Int(_), sym::simd_ctlz) => fx.bcx.ins().clz(lane),
                (ty::Uint(_) | ty::Int(_), sym::simd_ctpop) => fx.bcx.ins().popcnt(lane),
                (ty::Uint(_) | ty::Int(_), sym::simd_cttz) => fx.bcx.ins().ctz(lane),

                _ => unreachable!(),
            });
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
        | sym::simd_xor => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            // FIXME use vector instructions when possible
            simd_pair_for_each_lane(fx, x, y, ret, &|fx, lane_ty, _ret_lane_ty, x_lane, y_lane| {
                match (lane_ty.kind(), intrinsic) {
                    (ty::Uint(_), sym::simd_add) => fx.bcx.ins().iadd(x_lane, y_lane),
                    (ty::Uint(_), sym::simd_sub) => fx.bcx.ins().isub(x_lane, y_lane),
                    (ty::Uint(_), sym::simd_mul) => fx.bcx.ins().imul(x_lane, y_lane),
                    (ty::Uint(_), sym::simd_div) => fx.bcx.ins().udiv(x_lane, y_lane),
                    (ty::Uint(_), sym::simd_rem) => fx.bcx.ins().urem(x_lane, y_lane),

                    (ty::Int(_), sym::simd_add) => fx.bcx.ins().iadd(x_lane, y_lane),
                    (ty::Int(_), sym::simd_sub) => fx.bcx.ins().isub(x_lane, y_lane),
                    (ty::Int(_), sym::simd_mul) => fx.bcx.ins().imul(x_lane, y_lane),
                    (ty::Int(_), sym::simd_div) => fx.bcx.ins().sdiv(x_lane, y_lane),
                    (ty::Int(_), sym::simd_rem) => fx.bcx.ins().srem(x_lane, y_lane),

                    (ty::Float(_), sym::simd_add) => fx.bcx.ins().fadd(x_lane, y_lane),
                    (ty::Float(_), sym::simd_sub) => fx.bcx.ins().fsub(x_lane, y_lane),
                    (ty::Float(_), sym::simd_mul) => fx.bcx.ins().fmul(x_lane, y_lane),
                    (ty::Float(_), sym::simd_div) => fx.bcx.ins().fdiv(x_lane, y_lane),
                    (ty::Float(FloatTy::F32), sym::simd_rem) => fx.lib_call(
                        "fmodf",
                        vec![AbiParam::new(types::F32), AbiParam::new(types::F32)],
                        vec![AbiParam::new(types::F32)],
                        &[x_lane, y_lane],
                    )[0],
                    (ty::Float(FloatTy::F64), sym::simd_rem) => fx.lib_call(
                        "fmod",
                        vec![AbiParam::new(types::F64), AbiParam::new(types::F64)],
                        vec![AbiParam::new(types::F64)],
                        &[x_lane, y_lane],
                    )[0],

                    (ty::Uint(_), sym::simd_shl) => fx.bcx.ins().ishl(x_lane, y_lane),
                    (ty::Uint(_), sym::simd_shr) => fx.bcx.ins().ushr(x_lane, y_lane),
                    (ty::Uint(_), sym::simd_and) => fx.bcx.ins().band(x_lane, y_lane),
                    (ty::Uint(_), sym::simd_or) => fx.bcx.ins().bor(x_lane, y_lane),
                    (ty::Uint(_), sym::simd_xor) => fx.bcx.ins().bxor(x_lane, y_lane),

                    (ty::Int(_), sym::simd_shl) => fx.bcx.ins().ishl(x_lane, y_lane),
                    (ty::Int(_), sym::simd_shr) => fx.bcx.ins().sshr(x_lane, y_lane),
                    (ty::Int(_), sym::simd_and) => fx.bcx.ins().band(x_lane, y_lane),
                    (ty::Int(_), sym::simd_or) => fx.bcx.ins().bor(x_lane, y_lane),
                    (ty::Int(_), sym::simd_xor) => fx.bcx.ins().bxor(x_lane, y_lane),

                    _ => unreachable!(),
                }
            });
        }

        // FIXME: simd_relaxed_fma doesn't relax to non-fused multiply-add
        sym::simd_fma | sym::simd_relaxed_fma => {
            intrinsic_args!(fx, args => (a, b, c); intrinsic);

            if !a.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, a.layout().ty);
                return;
            }
            assert_eq!(a.layout(), b.layout());
            assert_eq!(a.layout(), c.layout());
            assert_eq!(a.layout(), ret.layout());

            let layout = a.layout();
            let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
            let res_lane_layout = fx.layout_of(lane_ty);

            for lane in 0..lane_count {
                let a_lane = a.value_lane(fx, lane).load_scalar(fx);
                let b_lane = b.value_lane(fx, lane).load_scalar(fx);
                let c_lane = c.value_lane(fx, lane).load_scalar(fx);

                let res_lane = fx.bcx.ins().fma(a_lane, b_lane, c_lane);
                let res_lane = CValue::by_val(res_lane, res_lane_layout);

                ret.place_lane(fx, lane).write_cvalue(fx, res_lane);
            }
        }

        sym::simd_fmin | sym::simd_fmax => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            if !x.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, x.layout().ty);
                return;
            }

            // FIXME use vector instructions when possible
            simd_pair_for_each_lane(fx, x, y, ret, &|fx, lane_ty, _ret_lane_ty, x_lane, y_lane| {
                match lane_ty.kind() {
                    ty::Float(_) => {}
                    _ => unreachable!("{:?}", lane_ty),
                }
                match intrinsic {
                    sym::simd_fmin => crate::num::codegen_float_min(fx, x_lane, y_lane),
                    sym::simd_fmax => crate::num::codegen_float_max(fx, x_lane, y_lane),
                    _ => unreachable!(),
                }
            });
        }

        sym::simd_fsin
        | sym::simd_fcos
        | sym::simd_fexp
        | sym::simd_fexp2
        | sym::simd_flog
        | sym::simd_flog10
        | sym::simd_flog2
        | sym::simd_round => {
            intrinsic_args!(fx, args => (a); intrinsic);

            if !a.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, a.layout().ty);
                return;
            }

            simd_for_each_lane(fx, a, ret, &|fx, lane_ty, _ret_lane_ty, lane| {
                let lane_ty = match lane_ty.kind() {
                    ty::Float(FloatTy::F32) => types::F32,
                    ty::Float(FloatTy::F64) => types::F64,
                    _ => unreachable!("{:?}", lane_ty),
                };
                let name = match (intrinsic, lane_ty) {
                    (sym::simd_fsin, types::F32) => "sinf",
                    (sym::simd_fsin, types::F64) => "sin",
                    (sym::simd_fcos, types::F32) => "cosf",
                    (sym::simd_fcos, types::F64) => "cos",
                    (sym::simd_fexp, types::F32) => "expf",
                    (sym::simd_fexp, types::F64) => "exp",
                    (sym::simd_fexp2, types::F32) => "exp2f",
                    (sym::simd_fexp2, types::F64) => "exp2",
                    (sym::simd_flog, types::F32) => "logf",
                    (sym::simd_flog, types::F64) => "log",
                    (sym::simd_flog10, types::F32) => "log10f",
                    (sym::simd_flog10, types::F64) => "log10",
                    (sym::simd_flog2, types::F32) => "log2f",
                    (sym::simd_flog2, types::F64) => "log2",
                    (sym::simd_round, types::F32) => "roundf",
                    (sym::simd_round, types::F64) => "round",
                    _ => unreachable!("{:?}", intrinsic),
                };
                fx.lib_call(
                    name,
                    vec![AbiParam::new(lane_ty)],
                    vec![AbiParam::new(lane_ty)],
                    &[lane],
                )[0]
            });
        }

        sym::simd_fabs | sym::simd_fsqrt | sym::simd_ceil | sym::simd_floor | sym::simd_trunc => {
            intrinsic_args!(fx, args => (a); intrinsic);

            if !a.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, a.layout().ty);
                return;
            }

            simd_for_each_lane(fx, a, ret, &|fx, lane_ty, _ret_lane_ty, lane| {
                match lane_ty.kind() {
                    ty::Float(_) => {}
                    _ => unreachable!("{:?}", lane_ty),
                }
                match intrinsic {
                    sym::simd_fabs => fx.bcx.ins().fabs(lane),
                    sym::simd_fsqrt => fx.bcx.ins().sqrt(lane),
                    sym::simd_ceil => fx.bcx.ins().ceil(lane),
                    sym::simd_floor => fx.bcx.ins().floor(lane),
                    sym::simd_trunc => fx.bcx.ins().trunc(lane),
                    _ => unreachable!(),
                }
            });
        }

        sym::simd_reduce_add_ordered => {
            intrinsic_args!(fx, args => (v, acc); intrinsic);
            let acc = acc.load_scalar(fx);

            // FIXME there must be no acc param for integer vectors
            if !v.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, v.layout().ty);
                return;
            }

            simd_reduce(fx, v, Some(acc), ret, &|fx, lane_ty, a, b| {
                if lane_ty.is_floating_point() {
                    fx.bcx.ins().fadd(a, b)
                } else {
                    fx.bcx.ins().iadd(a, b)
                }
            });
        }

        sym::simd_reduce_add_unordered => {
            intrinsic_args!(fx, args => (v); intrinsic);

            // FIXME there must be no acc param for integer vectors
            if !v.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, v.layout().ty);
                return;
            }

            simd_reduce(fx, v, None, ret, &|fx, lane_ty, a, b| {
                if lane_ty.is_floating_point() {
                    fx.bcx.ins().fadd(a, b)
                } else {
                    fx.bcx.ins().iadd(a, b)
                }
            });
        }

        sym::simd_reduce_mul_ordered => {
            intrinsic_args!(fx, args => (v, acc); intrinsic);
            let acc = acc.load_scalar(fx);

            // FIXME there must be no acc param for integer vectors
            if !v.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, v.layout().ty);
                return;
            }

            simd_reduce(fx, v, Some(acc), ret, &|fx, lane_ty, a, b| {
                if lane_ty.is_floating_point() {
                    fx.bcx.ins().fmul(a, b)
                } else {
                    fx.bcx.ins().imul(a, b)
                }
            });
        }

        sym::simd_reduce_mul_unordered => {
            intrinsic_args!(fx, args => (v); intrinsic);

            // FIXME there must be no acc param for integer vectors
            if !v.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, v.layout().ty);
                return;
            }

            simd_reduce(fx, v, None, ret, &|fx, lane_ty, a, b| {
                if lane_ty.is_floating_point() {
                    fx.bcx.ins().fmul(a, b)
                } else {
                    fx.bcx.ins().imul(a, b)
                }
            });
        }

        sym::simd_reduce_all => {
            intrinsic_args!(fx, args => (v); intrinsic);

            if !v.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, v.layout().ty);
                return;
            }

            simd_reduce_bool(fx, v, ret, &|fx, a, b| fx.bcx.ins().band(a, b));
        }

        sym::simd_reduce_any => {
            intrinsic_args!(fx, args => (v); intrinsic);

            if !v.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, v.layout().ty);
                return;
            }

            simd_reduce_bool(fx, v, ret, &|fx, a, b| fx.bcx.ins().bor(a, b));
        }

        sym::simd_reduce_and => {
            intrinsic_args!(fx, args => (v); intrinsic);

            if !v.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, v.layout().ty);
                return;
            }

            simd_reduce(fx, v, None, ret, &|fx, _ty, a, b| fx.bcx.ins().band(a, b));
        }

        sym::simd_reduce_or => {
            intrinsic_args!(fx, args => (v); intrinsic);

            if !v.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, v.layout().ty);
                return;
            }

            simd_reduce(fx, v, None, ret, &|fx, _ty, a, b| fx.bcx.ins().bor(a, b));
        }

        sym::simd_reduce_xor => {
            intrinsic_args!(fx, args => (v); intrinsic);

            if !v.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, v.layout().ty);
                return;
            }

            simd_reduce(fx, v, None, ret, &|fx, _ty, a, b| fx.bcx.ins().bxor(a, b));
        }

        sym::simd_reduce_min => {
            intrinsic_args!(fx, args => (v); intrinsic);

            if !v.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, v.layout().ty);
                return;
            }

            simd_reduce(fx, v, None, ret, &|fx, ty, a, b| {
                let lt = match ty.kind() {
                    ty::Int(_) => fx.bcx.ins().icmp(IntCC::SignedLessThan, a, b),
                    ty::Uint(_) => fx.bcx.ins().icmp(IntCC::UnsignedLessThan, a, b),
                    ty::Float(_) => return crate::num::codegen_float_min(fx, a, b),
                    _ => unreachable!(),
                };
                fx.bcx.ins().select(lt, a, b)
            });
        }

        sym::simd_reduce_max => {
            intrinsic_args!(fx, args => (v); intrinsic);

            if !v.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, v.layout().ty);
                return;
            }

            simd_reduce(fx, v, None, ret, &|fx, ty, a, b| {
                let gt = match ty.kind() {
                    ty::Int(_) => fx.bcx.ins().icmp(IntCC::SignedGreaterThan, a, b),
                    ty::Uint(_) => fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan, a, b),
                    ty::Float(_) => return crate::num::codegen_float_max(fx, a, b),
                    _ => unreachable!(),
                };
                fx.bcx.ins().select(gt, a, b)
            });
        }

        sym::simd_select => {
            intrinsic_args!(fx, args => (m, a, b); intrinsic);

            if !m.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, m.layout().ty);
                return;
            }
            if !a.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, a.layout().ty);
                return;
            }
            assert_eq!(a.layout(), b.layout());

            let (lane_count, lane_ty) = a.layout().ty.simd_size_and_type(fx.tcx);
            let lane_layout = fx.layout_of(lane_ty);

            for lane in 0..lane_count {
                let m_lane = m.value_lane(fx, lane).load_scalar(fx);
                let a_lane = a.value_lane(fx, lane).load_scalar(fx);
                let b_lane = b.value_lane(fx, lane).load_scalar(fx);

                let m_lane = fx.bcx.ins().icmp_imm(IntCC::Equal, m_lane, 0);
                let res_lane =
                    CValue::by_val(fx.bcx.ins().select(m_lane, b_lane, a_lane), lane_layout);

                ret.place_lane(fx, lane).write_cvalue(fx, res_lane);
            }
        }

        sym::simd_select_bitmask => {
            intrinsic_args!(fx, args => (m, a, b); intrinsic);

            if !a.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, a.layout().ty);
                return;
            }
            assert_eq!(a.layout(), b.layout());

            let (lane_count, lane_ty) = a.layout().ty.simd_size_and_type(fx.tcx);
            let lane_layout = fx.layout_of(lane_ty);

            let expected_int_bits = lane_count.max(8);
            let expected_bytes = expected_int_bits / 8 + ((expected_int_bits % 8 > 0) as u64);

            let m = match m.layout().ty.kind() {
                ty::Uint(i) if i.bit_width() == Some(expected_int_bits) => m.load_scalar(fx),
                ty::Array(elem, len)
                    if matches!(elem.kind(), ty::Uint(ty::UintTy::U8))
                        && len
                            .try_to_target_usize(fx.tcx)
                            .expect("expected monomorphic const in codegen")
                            == expected_bytes =>
                {
                    m.force_stack(fx).0.load(
                        fx,
                        Type::int(expected_int_bits as u16).unwrap(),
                        MemFlags::trusted(),
                    )
                }
                _ => {
                    fx.tcx.dcx().span_fatal(
                        span,
                        format!(
                            "invalid monomorphization of `simd_select_bitmask` intrinsic: \
                            cannot accept `{}` as mask, expected `u{}` or `[u8; {}]`",
                            ret.layout().ty,
                            expected_int_bits,
                            expected_bytes
                        ),
                    );
                }
            };

            for lane in 0..lane_count {
                // The bit order of the mask depends on the byte endianness, LSB-first for
                // little endian and MSB-first for big endian.
                let mask_lane = match fx.tcx.sess.target.endian {
                    Endian::Big => lane_count - 1 - lane,
                    Endian::Little => lane,
                };
                let m_lane = fx.bcx.ins().ushr_imm(m, u64::from(mask_lane) as i64);
                let m_lane = fx.bcx.ins().band_imm(m_lane, 1);
                let a_lane = a.value_lane(fx, lane).load_scalar(fx);
                let b_lane = b.value_lane(fx, lane).load_scalar(fx);

                let m_lane = fx.bcx.ins().icmp_imm(IntCC::Equal, m_lane, 0);
                let res_lane =
                    CValue::by_val(fx.bcx.ins().select(m_lane, b_lane, a_lane), lane_layout);

                ret.place_lane(fx, lane).write_cvalue(fx, res_lane);
            }
        }

        sym::simd_bitmask => {
            intrinsic_args!(fx, args => (a); intrinsic);

            let (lane_count, lane_ty) = a.layout().ty.simd_size_and_type(fx.tcx);
            let lane_clif_ty = fx.clif_type(lane_ty).unwrap();

            // The `fn simd_bitmask(vector) -> unsigned integer` intrinsic takes a
            // vector mask and returns the most significant bit (MSB) of each lane in the form
            // of either:
            // * an unsigned integer
            // * an array of `u8`
            // If the vector has less than 8 lanes, a u8 is returned with zeroed trailing bits.
            //
            // The bit order of the result depends on the byte endianness, LSB-first for little
            // endian and MSB-first for big endian.
            let expected_int_bits = lane_count.max(8);
            let expected_bytes = expected_int_bits / 8 + ((expected_int_bits % 8 > 0) as u64);

            match lane_ty.kind() {
                ty::Int(_) | ty::Uint(_) => {}
                _ => {
                    fx.tcx.dcx().span_fatal(
                        span,
                        format!(
                            "invalid monomorphization of `simd_bitmask` intrinsic: \
                            vector argument `{}`'s element type `{}`, expected integer element \
                            type",
                            a.layout().ty,
                            lane_ty
                        ),
                    );
                }
            }

            let res_type =
                Type::int_with_byte_size(u16::try_from(expected_bytes).unwrap()).unwrap();
            let mut res = type_zero_value(&mut fx.bcx, res_type);

            let lanes = match fx.tcx.sess.target.endian {
                Endian::Big => Box::new(0..lane_count) as Box<dyn Iterator<Item = u64>>,
                Endian::Little => Box::new((0..lane_count).rev()) as Box<dyn Iterator<Item = u64>>,
            };
            for lane in lanes {
                let a_lane = a.value_lane(fx, lane).load_scalar(fx);

                // extract sign bit of an int
                let a_lane_sign = fx.bcx.ins().ushr_imm(a_lane, i64::from(lane_clif_ty.bits() - 1));

                // shift sign bit into result
                let a_lane_sign = clif_intcast(fx, a_lane_sign, res_type, false);
                res = fx.bcx.ins().ishl_imm(res, 1);
                res = fx.bcx.ins().bor(res, a_lane_sign);
            }

            match ret.layout().ty.kind() {
                ty::Uint(i) if i.bit_width() == Some(expected_int_bits) => {}
                ty::Array(elem, len)
                    if matches!(elem.kind(), ty::Uint(ty::UintTy::U8))
                        && len
                            .try_to_target_usize(fx.tcx)
                            .expect("expected monomorphic const in codegen")
                            == expected_bytes => {}
                _ => {
                    fx.tcx.dcx().span_fatal(
                        span,
                        format!(
                            "invalid monomorphization of `simd_bitmask` intrinsic: \
                            cannot return `{}`, expected `u{}` or `[u8; {}]`",
                            ret.layout().ty,
                            expected_int_bits,
                            expected_bytes
                        ),
                    );
                }
            }

            let res = CValue::by_val(res, ret.layout());
            ret.write_cvalue(fx, res);
        }

        sym::simd_saturating_add | sym::simd_saturating_sub => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            let bin_op = match intrinsic {
                sym::simd_saturating_add => BinOp::Add,
                sym::simd_saturating_sub => BinOp::Sub,
                _ => unreachable!(),
            };

            // FIXME use vector instructions when possible
            simd_pair_for_each_lane_typed(fx, x, y, ret, &|fx, x_lane, y_lane| {
                crate::num::codegen_saturating_int_binop(fx, bin_op, x_lane, y_lane)
            });
        }

        sym::simd_expose_provenance | sym::simd_with_exposed_provenance | sym::simd_cast_ptr => {
            intrinsic_args!(fx, args => (arg); intrinsic);
            ret.write_cvalue_transmute(fx, arg);
        }

        sym::simd_arith_offset => {
            intrinsic_args!(fx, args => (ptr, offset); intrinsic);

            let (lane_count, ptr_lane_ty) = ptr.layout().ty.simd_size_and_type(fx.tcx);
            let pointee_ty = ptr_lane_ty.builtin_deref(true).unwrap();
            let pointee_size = fx.layout_of(pointee_ty).size.bytes();
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
            let ret_lane_layout = fx.layout_of(ret_lane_ty);
            assert_eq!(lane_count, ret_lane_count);

            for lane_idx in 0..lane_count {
                let ptr_lane = ptr.value_lane(fx, lane_idx).load_scalar(fx);
                let offset_lane = offset.value_lane(fx, lane_idx).load_scalar(fx);

                let ptr_diff = if pointee_size != 1 {
                    fx.bcx.ins().imul_imm(offset_lane, pointee_size as i64)
                } else {
                    offset_lane
                };
                let res_lane = fx.bcx.ins().iadd(ptr_lane, ptr_diff);
                let res_lane = CValue::by_val(res_lane, ret_lane_layout);

                ret.place_lane(fx, lane_idx).write_cvalue(fx, res_lane);
            }
        }

        sym::simd_masked_store => {
            intrinsic_args!(fx, args => (mask, ptr, val); intrinsic);

            let (val_lane_count, val_lane_ty) = val.layout().ty.simd_size_and_type(fx.tcx);
            let (mask_lane_count, _mask_lane_ty) = mask.layout().ty.simd_size_and_type(fx.tcx);
            assert_eq!(val_lane_count, mask_lane_count);
            let lane_clif_ty = fx.clif_type(val_lane_ty).unwrap();
            let ptr_val = ptr.load_scalar(fx);

            for lane_idx in 0..val_lane_count {
                let val_lane = val.value_lane(fx, lane_idx).load_scalar(fx);
                let mask_lane = mask.value_lane(fx, lane_idx).load_scalar(fx);

                let if_enabled = fx.bcx.create_block();
                let next = fx.bcx.create_block();

                fx.bcx.ins().brif(mask_lane, if_enabled, &[], next, &[]);
                fx.bcx.seal_block(if_enabled);

                fx.bcx.switch_to_block(if_enabled);
                let offset = lane_idx as i32 * lane_clif_ty.bytes() as i32;
                fx.bcx.ins().store(MemFlags::trusted(), val_lane, ptr_val, Offset32::new(offset));
                fx.bcx.ins().jump(next, &[]);

                fx.bcx.seal_block(next);
                fx.bcx.switch_to_block(next);

                fx.bcx.ins().nop();
            }
        }

        sym::simd_gather => {
            intrinsic_args!(fx, args => (val, ptr, mask); intrinsic);

            let (val_lane_count, val_lane_ty) = val.layout().ty.simd_size_and_type(fx.tcx);
            let (ptr_lane_count, _ptr_lane_ty) = ptr.layout().ty.simd_size_and_type(fx.tcx);
            let (mask_lane_count, _mask_lane_ty) = mask.layout().ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
            assert_eq!(val_lane_count, ptr_lane_count);
            assert_eq!(val_lane_count, mask_lane_count);
            assert_eq!(val_lane_count, ret_lane_count);

            let lane_clif_ty = fx.clif_type(val_lane_ty).unwrap();
            let ret_lane_layout = fx.layout_of(ret_lane_ty);

            for lane_idx in 0..ptr_lane_count {
                let val_lane = val.value_lane(fx, lane_idx).load_scalar(fx);
                let ptr_lane = ptr.value_lane(fx, lane_idx).load_scalar(fx);
                let mask_lane = mask.value_lane(fx, lane_idx).load_scalar(fx);

                let if_enabled = fx.bcx.create_block();
                let if_disabled = fx.bcx.create_block();
                let next = fx.bcx.create_block();
                let res_lane = fx.bcx.append_block_param(next, lane_clif_ty);

                fx.bcx.ins().brif(mask_lane, if_enabled, &[], if_disabled, &[]);
                fx.bcx.seal_block(if_enabled);
                fx.bcx.seal_block(if_disabled);

                fx.bcx.switch_to_block(if_enabled);
                let res = fx.bcx.ins().load(lane_clif_ty, MemFlags::trusted(), ptr_lane, 0);
                fx.bcx.ins().jump(next, &[res.into()]);

                fx.bcx.switch_to_block(if_disabled);
                fx.bcx.ins().jump(next, &[val_lane.into()]);

                fx.bcx.seal_block(next);
                fx.bcx.switch_to_block(next);

                fx.bcx.ins().nop();

                ret.place_lane(fx, lane_idx)
                    .write_cvalue(fx, CValue::by_val(res_lane, ret_lane_layout));
            }
        }

        sym::simd_masked_load => {
            intrinsic_args!(fx, args => (mask, ptr, val); intrinsic);

            let (val_lane_count, val_lane_ty) = val.layout().ty.simd_size_and_type(fx.tcx);
            let (mask_lane_count, _mask_lane_ty) = mask.layout().ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
            assert_eq!(val_lane_count, mask_lane_count);
            assert_eq!(val_lane_count, ret_lane_count);

            let lane_clif_ty = fx.clif_type(val_lane_ty).unwrap();
            let ret_lane_layout = fx.layout_of(ret_lane_ty);
            let ptr_val = ptr.load_scalar(fx);

            for lane_idx in 0..ret_lane_count {
                let val_lane = val.value_lane(fx, lane_idx).load_scalar(fx);
                let mask_lane = mask.value_lane(fx, lane_idx).load_scalar(fx);

                let if_enabled = fx.bcx.create_block();
                let if_disabled = fx.bcx.create_block();
                let next = fx.bcx.create_block();
                let res_lane = fx.bcx.append_block_param(next, lane_clif_ty);

                fx.bcx.ins().brif(mask_lane, if_enabled, &[], if_disabled, &[]);
                fx.bcx.seal_block(if_enabled);
                fx.bcx.seal_block(if_disabled);

                fx.bcx.switch_to_block(if_enabled);
                let offset = lane_idx as i32 * lane_clif_ty.bytes() as i32;
                let res = fx.bcx.ins().load(
                    lane_clif_ty,
                    MemFlags::trusted(),
                    ptr_val,
                    Offset32::new(offset),
                );
                fx.bcx.ins().jump(next, &[res.into()]);

                fx.bcx.switch_to_block(if_disabled);
                fx.bcx.ins().jump(next, &[val_lane.into()]);

                fx.bcx.seal_block(next);
                fx.bcx.switch_to_block(next);

                fx.bcx.ins().nop();

                ret.place_lane(fx, lane_idx)
                    .write_cvalue(fx, CValue::by_val(res_lane, ret_lane_layout));
            }
        }

        sym::simd_scatter => {
            intrinsic_args!(fx, args => (val, ptr, mask); intrinsic);

            let (val_lane_count, _val_lane_ty) = val.layout().ty.simd_size_and_type(fx.tcx);
            let (ptr_lane_count, _ptr_lane_ty) = ptr.layout().ty.simd_size_and_type(fx.tcx);
            let (mask_lane_count, _mask_lane_ty) = mask.layout().ty.simd_size_and_type(fx.tcx);
            assert_eq!(val_lane_count, ptr_lane_count);
            assert_eq!(val_lane_count, mask_lane_count);

            for lane_idx in 0..ptr_lane_count {
                let val_lane = val.value_lane(fx, lane_idx).load_scalar(fx);
                let ptr_lane = ptr.value_lane(fx, lane_idx).load_scalar(fx);
                let mask_lane = mask.value_lane(fx, lane_idx).load_scalar(fx);

                let if_enabled = fx.bcx.create_block();
                let next = fx.bcx.create_block();

                fx.bcx.ins().brif(mask_lane, if_enabled, &[], next, &[]);
                fx.bcx.seal_block(if_enabled);

                fx.bcx.switch_to_block(if_enabled);
                fx.bcx.ins().store(MemFlags::trusted(), val_lane, ptr_lane, 0);
                fx.bcx.ins().jump(next, &[]);

                fx.bcx.seal_block(next);
                fx.bcx.switch_to_block(next);
            }
        }

        _ => {
            fx.tcx.dcx().span_err(span, format!("Unknown SIMD intrinsic {}", intrinsic));
            // Prevent verifier error
            fx.bcx.ins().trap(TrapCode::user(1 /* unreachable */).unwrap());
            return;
        }
    }
    let ret_block = fx.get_block(target);
    fx.bcx.ins().jump(ret_block, &[]);
}
