//! Lowers intrinsic calls

use crate::{errors, MirPass};
use rustc_middle::mir::*;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;
use rustc_target::abi::{FieldIdx, VariantIdx};

pub struct LowerIntrinsics;

impl<'tcx> MirPass<'tcx> for LowerIntrinsics {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let local_decls = &body.local_decls;
        for block in body.basic_blocks.as_mut() {
            let terminator = block.terminator.as_mut().unwrap();
            if let TerminatorKind::Call { func, args, destination, target, .. } =
                &mut terminator.kind
            {
                let func_ty = func.ty(local_decls, tcx);
                let Some((intrinsic_name, substs)) = resolve_rust_intrinsic(tcx, func_ty) else {
                    continue;
                };
                match intrinsic_name {
                    sym::unreachable => {
                        terminator.kind = TerminatorKind::Unreachable;
                    }
                    sym::forget => {
                        if let Some(target) = *target {
                            block.statements.push(Statement {
                                source_info: terminator.source_info,
                                kind: StatementKind::Assign(Box::new((
                                    *destination,
                                    Rvalue::Use(Operand::Constant(Box::new(Constant {
                                        span: terminator.source_info.span,
                                        user_ty: None,
                                        literal: ConstantKind::zero_sized(tcx.types.unit),
                                    }))),
                                ))),
                            });
                            terminator.kind = TerminatorKind::Goto { target };
                        }
                    }
                    sym::copy_nonoverlapping => {
                        let target = target.unwrap();
                        let mut args = args.drain(..);
                        block.statements.push(Statement {
                            source_info: terminator.source_info,
                            kind: StatementKind::Intrinsic(Box::new(
                                NonDivergingIntrinsic::CopyNonOverlapping(
                                    rustc_middle::mir::CopyNonOverlapping {
                                        src: args.next().unwrap(),
                                        dst: args.next().unwrap(),
                                        count: args.next().unwrap(),
                                    },
                                ),
                            )),
                        });
                        assert_eq!(
                            args.next(),
                            None,
                            "Extra argument for copy_non_overlapping intrinsic"
                        );
                        drop(args);
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::assume => {
                        let target = target.unwrap();
                        let mut args = args.drain(..);
                        block.statements.push(Statement {
                            source_info: terminator.source_info,
                            kind: StatementKind::Intrinsic(Box::new(
                                NonDivergingIntrinsic::Assume(args.next().unwrap()),
                            )),
                        });
                        assert_eq!(
                            args.next(),
                            None,
                            "Extra argument for copy_non_overlapping intrinsic"
                        );
                        drop(args);
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::wrapping_add
                    | sym::wrapping_sub
                    | sym::wrapping_mul
                    | sym::unchecked_add
                    | sym::unchecked_sub
                    | sym::unchecked_mul
                    | sym::unchecked_div
                    | sym::unchecked_rem
                    | sym::unchecked_shl
                    | sym::unchecked_shr => {
                        let target = target.unwrap();
                        let lhs;
                        let rhs;
                        {
                            let mut args = args.drain(..);
                            lhs = args.next().unwrap();
                            rhs = args.next().unwrap();
                        }
                        let bin_op = match intrinsic_name {
                            sym::wrapping_add => BinOp::Add,
                            sym::wrapping_sub => BinOp::Sub,
                            sym::wrapping_mul => BinOp::Mul,
                            sym::unchecked_add => BinOp::AddUnchecked,
                            sym::unchecked_sub => BinOp::SubUnchecked,
                            sym::unchecked_mul => BinOp::MulUnchecked,
                            sym::unchecked_div => BinOp::Div,
                            sym::unchecked_rem => BinOp::Rem,
                            sym::unchecked_shl => BinOp::ShlUnchecked,
                            sym::unchecked_shr => BinOp::ShrUnchecked,
                            _ => bug!("unexpected intrinsic"),
                        };
                        block.statements.push(Statement {
                            source_info: terminator.source_info,
                            kind: StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::BinaryOp(bin_op, Box::new((lhs, rhs))),
                            ))),
                        });
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::add_with_overflow | sym::sub_with_overflow | sym::mul_with_overflow => {
                        if let Some(target) = *target {
                            let lhs;
                            let rhs;
                            {
                                let mut args = args.drain(..);
                                lhs = args.next().unwrap();
                                rhs = args.next().unwrap();
                            }
                            let bin_op = match intrinsic_name {
                                sym::add_with_overflow => BinOp::Add,
                                sym::sub_with_overflow => BinOp::Sub,
                                sym::mul_with_overflow => BinOp::Mul,
                                _ => bug!("unexpected intrinsic"),
                            };
                            block.statements.push(Statement {
                                source_info: terminator.source_info,
                                kind: StatementKind::Assign(Box::new((
                                    *destination,
                                    Rvalue::CheckedBinaryOp(bin_op, Box::new((lhs, rhs))),
                                ))),
                            });
                            terminator.kind = TerminatorKind::Goto { target };
                        }
                    }
                    sym::size_of | sym::min_align_of => {
                        if let Some(target) = *target {
                            let tp_ty = substs.type_at(0);
                            let null_op = match intrinsic_name {
                                sym::size_of => NullOp::SizeOf,
                                sym::min_align_of => NullOp::AlignOf,
                                _ => bug!("unexpected intrinsic"),
                            };
                            block.statements.push(Statement {
                                source_info: terminator.source_info,
                                kind: StatementKind::Assign(Box::new((
                                    *destination,
                                    Rvalue::NullaryOp(null_op, tp_ty),
                                ))),
                            });
                            terminator.kind = TerminatorKind::Goto { target };
                        }
                    }
                    sym::read_via_copy => {
                        let [arg] = args.as_slice() else {
                            span_bug!(terminator.source_info.span, "Wrong number of arguments");
                        };
                        let derefed_place =
                            if let Some(place) = arg.place() && let Some(local) = place.as_local() {
                                tcx.mk_place_deref(local.into())
                            } else {
                                span_bug!(terminator.source_info.span, "Only passing a local is supported");
                            };
                        terminator.kind = match *target {
                            None => {
                                // No target means this read something uninhabited,
                                // so it must be unreachable, and we don't need to
                                // preserve the assignment either.
                                TerminatorKind::Unreachable
                            }
                            Some(target) => {
                                block.statements.push(Statement {
                                    source_info: terminator.source_info,
                                    kind: StatementKind::Assign(Box::new((
                                        *destination,
                                        Rvalue::Use(Operand::Copy(derefed_place)),
                                    ))),
                                });
                                TerminatorKind::Goto { target }
                            }
                        }
                    }
                    sym::write_via_move => {
                        let target = target.unwrap();
                        let Ok([ptr, val]) = <[_; 2]>::try_from(std::mem::take(args)) else {
                            span_bug!(
                                terminator.source_info.span,
                                "Wrong number of arguments for write_via_move intrinsic",
                            );
                        };
                        let derefed_place =
                            if let Some(place) = ptr.place() && let Some(local) = place.as_local() {
                                tcx.mk_place_deref(local.into())
                            } else {
                                span_bug!(terminator.source_info.span, "Only passing a local is supported");
                            };
                        block.statements.push(Statement {
                            source_info: terminator.source_info,
                            kind: StatementKind::Assign(Box::new((
                                derefed_place,
                                Rvalue::Use(val),
                            ))),
                        });
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::discriminant_value => {
                        if let (Some(target), Some(arg)) = (*target, args[0].place()) {
                            let arg = tcx.mk_place_deref(arg);
                            block.statements.push(Statement {
                                source_info: terminator.source_info,
                                kind: StatementKind::Assign(Box::new((
                                    *destination,
                                    Rvalue::Discriminant(arg),
                                ))),
                            });
                            terminator.kind = TerminatorKind::Goto { target };
                        }
                    }
                    sym::offset => {
                        let target = target.unwrap();
                        let Ok([ptr, delta]) = <[_; 2]>::try_from(std::mem::take(args)) else {
                            span_bug!(
                                terminator.source_info.span,
                                "Wrong number of arguments for offset intrinsic",
                            );
                        };
                        block.statements.push(Statement {
                            source_info: terminator.source_info,
                            kind: StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::BinaryOp(BinOp::Offset, Box::new((ptr, delta))),
                            ))),
                        });
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::option_payload_ptr => {
                        if let (Some(target), Some(arg)) = (*target, args[0].place()) {
                            let ty::RawPtr(ty::TypeAndMut { ty: dest_ty, .. }) =
                                destination.ty(local_decls, tcx).ty.kind()
                            else { bug!(); };

                            block.statements.push(Statement {
                                source_info: terminator.source_info,
                                kind: StatementKind::Assign(Box::new((
                                    *destination,
                                    Rvalue::AddressOf(
                                        Mutability::Not,
                                        arg.project_deeper(
                                            &[
                                                PlaceElem::Deref,
                                                PlaceElem::Downcast(
                                                    Some(sym::Some),
                                                    VariantIdx::from_u32(1),
                                                ),
                                                PlaceElem::Field(FieldIdx::from_u32(0), *dest_ty),
                                            ],
                                            tcx,
                                        ),
                                    ),
                                ))),
                            });
                            terminator.kind = TerminatorKind::Goto { target };
                        }
                    }
                    sym::transmute | sym::transmute_unchecked => {
                        let dst_ty = destination.ty(local_decls, tcx).ty;
                        let Ok([arg]) = <[_; 1]>::try_from(std::mem::take(args)) else {
                            span_bug!(
                                terminator.source_info.span,
                                "Wrong number of arguments for transmute intrinsic",
                            );
                        };

                        // Always emit the cast, even if we transmute to an uninhabited type,
                        // because that lets CTFE and codegen generate better error messages
                        // when such a transmute actually ends up reachable.
                        block.statements.push(Statement {
                            source_info: terminator.source_info,
                            kind: StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::Cast(CastKind::Transmute, arg, dst_ty),
                            ))),
                        });

                        if let Some(target) = *target {
                            terminator.kind = TerminatorKind::Goto { target };
                        } else {
                            terminator.kind = TerminatorKind::Unreachable;
                        }
                    }
                    _ if intrinsic_name.as_str().starts_with("simd_shuffle") => {
                        validate_simd_shuffle(tcx, args, terminator.source_info.span);
                    }
                    _ => {}
                }
            }
        }
    }
}

fn resolve_rust_intrinsic<'tcx>(
    tcx: TyCtxt<'tcx>,
    func_ty: Ty<'tcx>,
) -> Option<(Symbol, SubstsRef<'tcx>)> {
    if let ty::FnDef(def_id, substs) = *func_ty.kind() {
        if tcx.is_intrinsic(def_id) {
            return Some((tcx.item_name(def_id), substs));
        }
    }
    None
}

fn validate_simd_shuffle<'tcx>(tcx: TyCtxt<'tcx>, args: &[Operand<'tcx>], span: Span) {
    if !matches!(args[2], Operand::Constant(_)) {
        tcx.sess.emit_err(errors::SimdShuffleLastConst { span });
    }
}
