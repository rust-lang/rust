//! A pass that checks for and simplifies calls to `pow` where the receiver is a power of
//! two. This can be done with `<<` instead.

use crate::MirPass;
use rustc_const_eval::interpret::{ConstValue, Scalar};
use rustc_hir::definitions::{DefPathData, DisambiguatedDefPathData};
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt, UintTy};
use rustc_span::sym;
use rustc_target::abi::FieldIdx;

pub struct SimplifyPowOfTwo;

impl<'tcx> MirPass<'tcx> for SimplifyPowOfTwo {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut patch = MirPatch::new(body);

        for (i, bb) in body.basic_blocks.iter_enumerated() {
            let term = bb.terminator();
            let source_info = term.source_info;
            let span = source_info.span;

            if let TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    target: Some(target),
                    unwind,
                    call_source: CallSource::Normal,
                    ..
                } = &term.kind
                && let Some(def_id) = func.const_fn_def().map(|def| def.0)
                && let def_path = tcx.def_path(def_id)
                && tcx.crate_name(def_path.krate) == sym::core
                // FIXME(Centri3): I feel like we should do this differently...
                && let [
                    DisambiguatedDefPathData { data: DefPathData::TypeNs(sym::num), disambiguator: 0 },
                    DisambiguatedDefPathData { data: DefPathData::Impl, .. },
                    DisambiguatedDefPathData { data: DefPathData::ValueNs(sym::pow), .. },
                ] = &*def_path.data
                && let [recv, exp] = args.as_slice()
                && let Some(recv_const) = recv.constant()
                && let ConstantKind::Val(
                    ConstValue::Scalar(Scalar::Int(recv_int)),
                    recv_ty,
                ) = recv_const.literal
                && let Ok(recv_val) = match recv_ty.kind() {
                    ty::Int(_) => {
                        let result = recv_int.try_to_int(recv_int.size()).unwrap_or(-1).max(0);
                        if result > 0 {
                            Ok(result as u128)
                        } else {
                            continue;
                        }
                    },
                    ty::Uint(_) => recv_int.try_to_uint(recv_int.size()),
                    _ => continue,
                }
                && let power_used = f32::log2(recv_val as f32)
                // Precision loss means it's not a power of two
                && power_used == (power_used as u32) as f32
                // `0` would be `1.pow()`, which we shouldn't try to optimize as it's
                // already entirely optimized away
                && power_used != 0.0
                // Same here
                && recv_val != 0
            {
                let power_used = power_used as u32;
                let loc = Location { block: i, statement_index: bb.statements.len() };
                let exp_ty = Ty::new(tcx, ty::Uint(UintTy::U32));
                let checked_mul =
                    patch.new_temp(Ty::new_tup(tcx, &[exp_ty, Ty::new_bool(tcx)]), span);

                // If this is not `2.pow(...)`, we need to multiply the number of times we
                // shift the bits left by the receiver's power of two used, e.g.:
                //
                // > 2 -> 1
                // > 4 -> 2
                // > 16 -> 4
                // > 256 -> 8
                //
                // If this is `1`, then we *could* remove this entirely but it'll be
                // optimized out anyway by later passes (or perhaps LLVM) so it's entirely
                // unnecessary to do so.
                patch.add_assign(
                    loc,
                    checked_mul.into(),
                    Rvalue::CheckedBinaryOp(
                        BinOp::Mul,
                        Box::new((
                            exp.clone(),
                            Operand::Constant(Box::new(Constant {
                                span,
                                user_ty: None,
                                literal: ConstantKind::Val(
                                    ConstValue::from_u32(power_used),
                                    exp_ty,
                                ),
                            })),
                        )),
                    ),
                );

                let num_shl = tcx.mk_place_field(checked_mul.into(), FieldIdx::from_u32(0), exp_ty);
                let mul_result = tcx.mk_place_field(
                    checked_mul.into(),
                    FieldIdx::from_u32(1),
                    Ty::new_bool(tcx),
                );
                let shl_result = patch.new_temp(Ty::new_bool(tcx), span);

                // Whether the shl will overflow, if so we return 0
                patch.add_assign(
                    loc,
                    shl_result.into(),
                    Rvalue::BinaryOp(
                        BinOp::Lt,
                        Box::new((
                            Operand::Copy(num_shl),
                            Operand::Constant(Box::new(Constant {
                                span,
                                user_ty: None,
                                literal: ConstantKind::Val(
                                    ConstValue::from_u32(recv_int.size().bits() as u32),
                                    exp_ty,
                                ),
                            })),
                        )),
                    ),
                );

                let should_be_zero_bool = patch.new_temp(Ty::new_bool(tcx), span);
                let should_be_zero = patch.new_temp(recv_ty, span);

                patch.add_assign(
                    loc,
                    should_be_zero_bool.into(),
                    Rvalue::BinaryOp(
                        BinOp::BitOr,
                        Box::new((
                            Operand::Copy(mul_result.into()),
                            Operand::Copy(shl_result.into()),
                        )),
                    ),
                );

                patch.add_assign(
                    loc,
                    should_be_zero.into(),
                    Rvalue::Cast(
                        CastKind::IntToInt,
                        Operand::Copy(should_be_zero_bool.into()),
                        recv_ty,
                    ),
                );

                let shl_exp_ty = patch.new_temp(exp_ty, span);
                let shl = patch.new_temp(recv_ty, span);

                patch.add_assign(
                    loc,
                    shl_exp_ty.into(),
                    Rvalue::BinaryOp(
                        BinOp::Shl,
                        Box::new((
                            Operand::Constant(Box::new(Constant {
                                span,
                                user_ty: None,
                                literal: ConstantKind::Val(ConstValue::from_u32(1), exp_ty),
                            })),
                            Operand::Copy(num_shl.into()),
                        )),
                    ),
                );

                patch.add_assign(
                    loc,
                    shl.into(),
                    Rvalue::Cast(CastKind::IntToInt, Operand::Copy(shl_exp_ty.into()), recv_ty),
                );

                patch.add_assign(
                    loc,
                    *destination,
                    Rvalue::BinaryOp(
                        BinOp::MulUnchecked,
                        Box::new((Operand::Copy(shl.into()), Operand::Copy(should_be_zero.into()))),
                    ),
                );

                // shl doesn't set the overflow flag on x86_64 or even in Rust, so shr to
                // see if it overflowed. If it equals 1, it did not, but we also need to
                // check `shl_result` to ensure that if this is a multiple of the type's
                // size it won't wrap back over to 1
                //
                // FIXME(Centri3): Do we use `debug_assertions` or `overflow_checks` here?
                if tcx.sess.opts.debug_assertions {
                    let shr = patch.new_temp(recv_ty, span);
                    let shl_eq_shr = patch.new_temp(Ty::new_bool(tcx), span);
                    let overflowed = patch.new_temp(Ty::new_bool(tcx), span);

                    patch.add_assign(
                        loc,
                        shr.into(),
                        Rvalue::BinaryOp(
                            BinOp::Shr,
                            Box::new((Operand::Copy(shl.into()), Operand::Copy(num_shl.into()))),
                        ),
                    );

                    patch.add_assign(
                        loc,
                        shl_eq_shr.into(),
                        Rvalue::BinaryOp(
                            BinOp::Eq,
                            Box::new((Operand::Copy(shl.into()), Operand::Copy(shr.into()))),
                        ),
                    );

                    patch.add_assign(
                        loc,
                        overflowed.into(),
                        Rvalue::BinaryOp(
                            BinOp::BitAnd,
                            Box::new((
                                Operand::Copy(shl_eq_shr.into()),
                                Operand::Copy(shl_result.into()),
                            )),
                        ),
                    );

                    patch.patch_terminator(
                        i,
                        TerminatorKind::Assert {
                            cond: Operand::Copy(overflowed.into()),
                            expected: true,
                            msg: Box::new(AssertMessage::Overflow(
                                // For consistency with the previous error message, though
                                // it's technically incorrect
                                BinOp::Mul,
                                Operand::Constant(Box::new(Constant {
                                    span,
                                    user_ty: None,
                                    literal: ConstantKind::Val(
                                        ConstValue::Scalar(Scalar::from_u32(1)),
                                        exp_ty,
                                    ),
                                })),
                                Operand::Copy(num_shl.into()),
                            )),
                            target: *target,
                            unwind: *unwind,
                        },
                    );
                } else {
                    patch.patch_terminator(i, TerminatorKind::Goto { target: *target });
                }
            }
        }

        patch.apply(body);
    }
}
