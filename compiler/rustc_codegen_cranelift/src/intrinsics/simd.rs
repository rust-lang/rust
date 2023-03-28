//! Codegen `extern "platform-intrinsic"` intrinsics.

use rustc_middle::ty::subst::SubstsRef;
use rustc_span::Symbol;
use rustc_target::abi::Endian;

use super::*;
use crate::prelude::*;

fn report_simd_type_validation_error(
    fx: &mut FunctionCx<'_, '_, '_>,
    intrinsic: Symbol,
    span: Span,
    ty: Ty<'_>,
) {
    fx.tcx.sess.span_err(span, &format!("invalid monomorphization of `{}` intrinsic: expected SIMD input type, found non-SIMD `{}`", intrinsic, ty));
    // Prevent verifier error
    fx.bcx.ins().trap(TrapCode::UnreachableCodeReached);
}

pub(super) fn codegen_simd_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    intrinsic: Symbol,
    _substs: SubstsRef<'tcx>,
    args: &[mir::Operand<'tcx>],
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

        // simd_shuffle32<T, U>(x: T, y: T, idx: [u32; 32]) -> U
        _ if intrinsic.as_str().starts_with("simd_shuffle") => {
            let (x, y, idx) = match args {
                [x, y, idx] => (x, y, idx),
                _ => {
                    bug!("wrong number of args for intrinsic {intrinsic}");
                }
            };
            let x = codegen_operand(fx, x);
            let y = codegen_operand(fx, y);

            if !x.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, x.layout().ty);
                return;
            }

            // If this intrinsic is the older "simd_shuffleN" form, simply parse the integer.
            // If there is no suffix, use the index array length.
            let n: u16 = if intrinsic == sym::simd_shuffle {
                // Make sure this is actually an array, since typeck only checks the length-suffixed
                // version of this intrinsic.
                let idx_ty = fx.monomorphize(idx.ty(fx.mir, fx.tcx));
                match idx_ty.kind() {
                    ty::Array(ty, len) if matches!(ty.kind(), ty::Uint(ty::UintTy::U32)) => len
                        .try_eval_target_usize(fx.tcx, ty::ParamEnv::reveal_all())
                        .unwrap_or_else(|| {
                            span_bug!(span, "could not evaluate shuffle index array length")
                        })
                        .try_into()
                        .unwrap(),
                    _ => {
                        fx.tcx.sess.span_err(
                            span,
                            &format!(
                                "simd_shuffle index must be an array of `u32`, got `{}`",
                                idx_ty,
                            ),
                        );
                        // Prevent verifier error
                        fx.bcx.ins().trap(TrapCode::UnreachableCodeReached);
                        return;
                    }
                }
            } else {
                // FIXME remove this case
                intrinsic.as_str()["simd_shuffle".len()..].parse().unwrap()
            };

            assert_eq!(x.layout(), y.layout());
            let layout = x.layout();

            let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);

            assert_eq!(lane_ty, ret_lane_ty);
            assert_eq!(u64::from(n), ret_lane_count);

            let total_len = lane_count * 2;

            let indexes = {
                use rustc_middle::mir::interpret::*;
                let idx_const = crate::constant::mir_operand_get_const_val(fx, idx)
                    .expect("simd_shuffle* idx not const");

                let idx_bytes = match idx_const {
                    ConstValue::ByRef { alloc, offset } => {
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
            let base = codegen_operand(fx, base);
            let val = codegen_operand(fx, val);

            // FIXME validate
            let idx_const = if let Some(idx_const) =
                crate::constant::mir_operand_get_const_val(fx, idx)
            {
                idx_const
            } else {
                fx.tcx.sess.span_fatal(span, "Index argument for `simd_insert` is not a constant");
            };

            let idx = idx_const
                .try_to_bits(Size::from_bytes(4 /* u32*/))
                .unwrap_or_else(|| panic!("kind not scalar: {:?}", idx_const));
            let (lane_count, _lane_ty) = base.layout().ty.simd_size_and_type(fx.tcx);
            if idx >= lane_count.into() {
                fx.tcx.sess.span_fatal(
                    fx.mir.span,
                    &format!("[simd_insert] idx {} >= lane_count {}", idx, lane_count),
                );
            }

            ret.write_cvalue(fx, base);
            let ret_lane = ret.place_field(fx, FieldIdx::new(idx.try_into().unwrap()));
            ret_lane.write_cvalue(fx, val);
        }

        sym::simd_extract => {
            let (v, idx) = match args {
                [v, idx] => (v, idx),
                _ => {
                    bug!("wrong number of args for intrinsic {intrinsic}");
                }
            };
            let v = codegen_operand(fx, v);

            if !v.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, v.layout().ty);
                return;
            }

            let idx_const = if let Some(idx_const) =
                crate::constant::mir_operand_get_const_val(fx, idx)
            {
                idx_const
            } else {
                fx.tcx.sess.span_warn(span, "Index argument for `simd_extract` is not a constant");
                let trap_block = fx.bcx.create_block();
                let true_ = fx.bcx.ins().iconst(types::I8, 1);
                let ret_block = fx.get_block(target);
                fx.bcx.ins().brif(true_, trap_block, &[], ret_block, &[]);
                fx.bcx.switch_to_block(trap_block);
                crate::trap::trap_unimplemented(
                    fx,
                    "Index argument for `simd_extract` is not a constant",
                );
                return;
            };

            let idx = idx_const
                .try_to_bits(Size::from_bytes(4 /* u32*/))
                .unwrap_or_else(|| panic!("kind not scalar: {:?}", idx_const));
            let (lane_count, _lane_ty) = v.layout().ty.simd_size_and_type(fx.tcx);
            if idx >= lane_count.into() {
                fx.tcx.sess.span_fatal(
                    fx.mir.span,
                    &format!("[simd_extract] idx {} >= lane_count {}", idx, lane_count),
                );
            }

            let ret_lane = v.value_lane(fx, idx.try_into().unwrap());
            ret.write_cvalue(fx, ret_lane);
        }

        sym::simd_neg => {
            intrinsic_args!(fx, args => (a); intrinsic);

            if !a.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, a.layout().ty);
                return;
            }

            simd_for_each_lane(
                fx,
                a,
                ret,
                &|fx, lane_ty, _ret_lane_ty, lane| match lane_ty.kind() {
                    ty::Int(_) => fx.bcx.ins().ineg(lane),
                    ty::Float(_) => fx.bcx.ins().fneg(lane),
                    _ => unreachable!(),
                },
            );
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

        sym::simd_fma => {
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

        sym::simd_round => {
            intrinsic_args!(fx, args => (a); intrinsic);

            if !a.layout().ty.is_simd() {
                report_simd_type_validation_error(fx, intrinsic, span, a.layout().ty);
                return;
            }

            simd_for_each_lane(
                fx,
                a,
                ret,
                &|fx, lane_ty, _ret_lane_ty, lane| match lane_ty.kind() {
                    ty::Float(FloatTy::F32) => fx.lib_call(
                        "roundf",
                        vec![AbiParam::new(types::F32)],
                        vec![AbiParam::new(types::F32)],
                        &[lane],
                    )[0],
                    ty::Float(FloatTy::F64) => fx.lib_call(
                        "round",
                        vec![AbiParam::new(types::F64)],
                        vec![AbiParam::new(types::F64)],
                        &[lane],
                    )[0],
                    _ => unreachable!("{:?}", lane_ty),
                },
            );
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

        sym::simd_reduce_add_ordered | sym::simd_reduce_add_unordered => {
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

        sym::simd_reduce_mul_ordered | sym::simd_reduce_mul_unordered => {
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

            let m = m.load_scalar(fx);

            for lane in 0..lane_count {
                let m_lane = fx.bcx.ins().ushr_imm(m, u64::from(lane) as i64);
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
                    fx.tcx.sess.span_fatal(
                        span,
                        &format!(
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
                        && len.try_eval_target_usize(fx.tcx, ty::ParamEnv::reveal_all())
                            == Some(expected_bytes) => {}
                _ => {
                    fx.tcx.sess.span_fatal(
                        span,
                        &format!(
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

        sym::simd_expose_addr | sym::simd_from_exposed_addr | sym::simd_cast_ptr => {
            intrinsic_args!(fx, args => (arg); intrinsic);
            ret.write_cvalue_transmute(fx, arg);
        }

        sym::simd_arith_offset => {
            intrinsic_args!(fx, args => (ptr, offset); intrinsic);

            let (lane_count, ptr_lane_ty) = ptr.layout().ty.simd_size_and_type(fx.tcx);
            let pointee_ty = ptr_lane_ty.builtin_deref(true).unwrap().ty;
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
                fx.bcx.ins().jump(next, &[res]);

                fx.bcx.switch_to_block(if_disabled);
                fx.bcx.ins().jump(next, &[val_lane]);

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
            fx.tcx.sess.span_err(span, &format!("Unknown SIMD intrinsic {}", intrinsic));
            // Prevent verifier error
            fx.bcx.ins().trap(TrapCode::UnreachableCodeReached);
        }
    }
    let ret_block = fx.get_block(target);
    fx.bcx.ins().jump(ret_block, &[]);
}
