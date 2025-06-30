//! Lowers intrinsic calls

use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_span::sym;

use crate::take_array;

pub(super) struct LowerIntrinsics;

impl<'tcx> crate::MirPass<'tcx> for LowerIntrinsics {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let local_decls = &body.local_decls;
        for block in body.basic_blocks.as_mut() {
            let terminator = block.terminator.as_mut().unwrap();
            if let TerminatorKind::Call { func, args, destination, target, .. } =
                &mut terminator.kind
                && let ty::FnDef(def_id, generic_args) = *func.ty(local_decls, tcx).kind()
                && let Some(intrinsic) = tcx.intrinsic(def_id)
            {
                match intrinsic.name {
                    sym::unreachable => {
                        terminator.kind = TerminatorKind::Unreachable;
                    }
                    sym::ub_checks => {
                        let target = target.unwrap();
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::NullaryOp(NullOp::UbChecks, tcx.types.bool),
                            ))),
                        ));
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::contract_checks => {
                        let target = target.unwrap();
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::NullaryOp(NullOp::ContractChecks, tcx.types.bool),
                            ))),
                        ));
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::forget => {
                        let target = target.unwrap();
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::Use(Operand::Constant(Box::new(ConstOperand {
                                    span: terminator.source_info.span,
                                    user_ty: None,
                                    const_: Const::zero_sized(tcx.types.unit),
                                }))),
                            ))),
                        ));
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::copy_nonoverlapping => {
                        let target = target.unwrap();
                        let Ok([src, dst, count]) = take_array(args) else {
                            bug!("Wrong arguments for copy_non_overlapping intrinsic");
                        };
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Intrinsic(Box::new(
                                NonDivergingIntrinsic::CopyNonOverlapping(
                                    rustc_middle::mir::CopyNonOverlapping {
                                        src: src.node,
                                        dst: dst.node,
                                        count: count.node,
                                    },
                                ),
                            )),
                        ));
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::assume => {
                        let target = target.unwrap();
                        let Ok([arg]) = take_array(args) else {
                            bug!("Wrong arguments for assume intrinsic");
                        };
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Intrinsic(Box::new(NonDivergingIntrinsic::Assume(
                                arg.node,
                            ))),
                        ));
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::wrapping_add
                    | sym::wrapping_sub
                    | sym::wrapping_mul
                    | sym::three_way_compare
                    | sym::unchecked_add
                    | sym::unchecked_sub
                    | sym::unchecked_mul
                    | sym::unchecked_div
                    | sym::unchecked_rem
                    | sym::unchecked_shl
                    | sym::unchecked_shr => {
                        let target = target.unwrap();
                        let Ok([lhs, rhs]) = take_array(args) else {
                            bug!("Wrong arguments for {} intrinsic", intrinsic.name);
                        };
                        let bin_op = match intrinsic.name {
                            sym::wrapping_add => BinOp::Add,
                            sym::wrapping_sub => BinOp::Sub,
                            sym::wrapping_mul => BinOp::Mul,
                            sym::three_way_compare => BinOp::Cmp,
                            sym::unchecked_add => BinOp::AddUnchecked,
                            sym::unchecked_sub => BinOp::SubUnchecked,
                            sym::unchecked_mul => BinOp::MulUnchecked,
                            sym::unchecked_div => BinOp::Div,
                            sym::unchecked_rem => BinOp::Rem,
                            sym::unchecked_shl => BinOp::ShlUnchecked,
                            sym::unchecked_shr => BinOp::ShrUnchecked,
                            _ => bug!("unexpected intrinsic"),
                        };
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::BinaryOp(bin_op, Box::new((lhs.node, rhs.node))),
                            ))),
                        ));
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::add_with_overflow | sym::sub_with_overflow | sym::mul_with_overflow => {
                        let target = target.unwrap();
                        let Ok([lhs, rhs]) = take_array(args) else {
                            bug!("Wrong arguments for {} intrinsic", intrinsic.name);
                        };
                        let bin_op = match intrinsic.name {
                            sym::add_with_overflow => BinOp::AddWithOverflow,
                            sym::sub_with_overflow => BinOp::SubWithOverflow,
                            sym::mul_with_overflow => BinOp::MulWithOverflow,
                            _ => bug!("unexpected intrinsic"),
                        };
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::BinaryOp(bin_op, Box::new((lhs.node, rhs.node))),
                            ))),
                        ));
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::size_of | sym::align_of => {
                        let target = target.unwrap();
                        let tp_ty = generic_args.type_at(0);
                        let null_op = match intrinsic.name {
                            sym::size_of => NullOp::SizeOf,
                            sym::align_of => NullOp::AlignOf,
                            _ => bug!("unexpected intrinsic"),
                        };
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::NullaryOp(null_op, tp_ty),
                            ))),
                        ));
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::read_via_copy => {
                        let Ok([arg]) = take_array(args) else {
                            span_bug!(terminator.source_info.span, "Wrong number of arguments");
                        };
                        let derefed_place = if let Some(place) = arg.node.place()
                            && let Some(local) = place.as_local()
                        {
                            tcx.mk_place_deref(local.into())
                        } else {
                            span_bug!(
                                terminator.source_info.span,
                                "Only passing a local is supported"
                            );
                        };
                        // Add new statement at the end of the block that does the read, and patch
                        // up the terminator.
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::Use(Operand::Copy(derefed_place)),
                            ))),
                        ));
                        terminator.kind = match *target {
                            None => {
                                // No target means this read something uninhabited,
                                // so it must be unreachable.
                                TerminatorKind::Unreachable
                            }
                            Some(target) => TerminatorKind::Goto { target },
                        }
                    }
                    sym::write_via_move => {
                        let target = target.unwrap();
                        let Ok([ptr, val]) = take_array(args) else {
                            span_bug!(
                                terminator.source_info.span,
                                "Wrong number of arguments for write_via_move intrinsic",
                            );
                        };
                        let derefed_place = if let Some(place) = ptr.node.place()
                            && let Some(local) = place.as_local()
                        {
                            tcx.mk_place_deref(local.into())
                        } else {
                            span_bug!(
                                terminator.source_info.span,
                                "Only passing a local is supported"
                            );
                        };
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Assign(Box::new((derefed_place, Rvalue::Use(val.node)))),
                        ));
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::discriminant_value => {
                        let target = target.unwrap();
                        let Ok([arg]) = take_array(args) else {
                            span_bug!(
                                terminator.source_info.span,
                                "Wrong arguments for discriminant_value intrinsic"
                            );
                        };
                        let arg = arg.node.place().unwrap();
                        let arg = tcx.mk_place_deref(arg);
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::Discriminant(arg),
                            ))),
                        ));
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::offset => {
                        let target = target.unwrap();
                        let Ok([ptr, delta]) = take_array(args) else {
                            span_bug!(
                                terminator.source_info.span,
                                "Wrong number of arguments for offset intrinsic",
                            );
                        };
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::BinaryOp(BinOp::Offset, Box::new((ptr.node, delta.node))),
                            ))),
                        ));
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::slice_get_unchecked => {
                        let target = target.unwrap();
                        let Ok([ptrish, index]) = take_array(args) else {
                            span_bug!(
                                terminator.source_info.span,
                                "Wrong number of arguments for {intrinsic:?}",
                            );
                        };

                        let place = ptrish.node.place().unwrap();
                        assert!(!place.is_indirect());
                        let updated_place = place.project_deeper(
                            &[
                                ProjectionElem::Deref,
                                ProjectionElem::Index(
                                    index.node.place().unwrap().as_local().unwrap(),
                                ),
                            ],
                            tcx,
                        );

                        let ret_ty = generic_args.type_at(0);
                        let rvalue = match *ret_ty.kind() {
                            ty::RawPtr(_, Mutability::Not) => {
                                Rvalue::RawPtr(RawPtrKind::Const, updated_place)
                            }
                            ty::RawPtr(_, Mutability::Mut) => {
                                Rvalue::RawPtr(RawPtrKind::Mut, updated_place)
                            }
                            ty::Ref(region, _, Mutability::Not) => {
                                Rvalue::Ref(region, BorrowKind::Shared, updated_place)
                            }
                            ty::Ref(region, _, Mutability::Mut) => Rvalue::Ref(
                                region,
                                BorrowKind::Mut { kind: MutBorrowKind::Default },
                                updated_place,
                            ),
                            _ => bug!("Unknown return type {ret_ty:?}"),
                        };

                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Assign(Box::new((*destination, rvalue))),
                        ));
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::transmute | sym::transmute_unchecked => {
                        let dst_ty = destination.ty(local_decls, tcx).ty;
                        let Ok([arg]) = take_array(args) else {
                            span_bug!(
                                terminator.source_info.span,
                                "Wrong number of arguments for transmute intrinsic",
                            );
                        };

                        // Always emit the cast, even if we transmute to an uninhabited type,
                        // because that lets CTFE and codegen generate better error messages
                        // when such a transmute actually ends up reachable.
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::Cast(CastKind::Transmute, arg.node, dst_ty),
                            ))),
                        ));
                        if let Some(target) = *target {
                            terminator.kind = TerminatorKind::Goto { target };
                        } else {
                            terminator.kind = TerminatorKind::Unreachable;
                        }
                    }
                    sym::aggregate_raw_ptr => {
                        let Ok([data, meta]) = take_array(args) else {
                            span_bug!(
                                terminator.source_info.span,
                                "Wrong number of arguments for aggregate_raw_ptr intrinsic",
                            );
                        };
                        let target = target.unwrap();
                        let pointer_ty = generic_args.type_at(0);
                        let kind = if let ty::RawPtr(pointee_ty, mutability) = pointer_ty.kind() {
                            AggregateKind::RawPtr(*pointee_ty, *mutability)
                        } else {
                            span_bug!(
                                terminator.source_info.span,
                                "Return type of aggregate_raw_ptr intrinsic must be a raw pointer",
                            );
                        };
                        let fields = [data.node, meta.node];
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::Aggregate(Box::new(kind), fields.into()),
                            ))),
                        ));
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    sym::ptr_metadata => {
                        let Ok([ptr]) = take_array(args) else {
                            span_bug!(
                                terminator.source_info.span,
                                "Wrong number of arguments for ptr_metadata intrinsic",
                            );
                        };
                        let target = target.unwrap();
                        block.statements.push(Statement::new(
                            terminator.source_info,
                            StatementKind::Assign(Box::new((
                                *destination,
                                Rvalue::UnaryOp(UnOp::PtrMetadata, ptr.node),
                            ))),
                        ));
                        terminator.kind = TerminatorKind::Goto { target };
                    }
                    _ => {}
                }
            }
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}
