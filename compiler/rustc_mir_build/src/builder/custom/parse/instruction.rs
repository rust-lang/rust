use rustc_abi::{FieldIdx, VariantIdx};
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::*;
use rustc_middle::thir::*;
use rustc_middle::ty;
use rustc_middle::ty::cast::mir_cast_kind;
use rustc_span::Span;
use rustc_span::source_map::Spanned;

use super::{PResult, ParseCtxt, parse_by_kind};
use crate::builder::custom::ParseError;
use crate::builder::expr::as_constant::as_constant_inner;

impl<'a, 'tcx> ParseCtxt<'a, 'tcx> {
    pub(crate) fn parse_statement(&self, expr_id: ExprId) -> PResult<StatementKind<'tcx>> {
        parse_by_kind!(self, expr_id, _, "statement",
            @call(mir_storage_live, args) => {
                Ok(StatementKind::StorageLive(self.parse_local(args[0])?))
            },
            @call(mir_storage_dead, args) => {
                Ok(StatementKind::StorageDead(self.parse_local(args[0])?))
            },
            @call(mir_assume, args) => {
                let op = self.parse_operand(args[0])?;
                Ok(StatementKind::Intrinsic(Box::new(NonDivergingIntrinsic::Assume(op))))
            },
            @call(mir_deinit, args) => {
                Ok(StatementKind::Deinit(Box::new(self.parse_place(args[0])?)))
            },
            @call(mir_retag, args) => {
                Ok(StatementKind::Retag(RetagKind::Default, Box::new(self.parse_place(args[0])?)))
            },
            @call(mir_set_discriminant, args) => {
                let place = self.parse_place(args[0])?;
                let var = self.parse_integer_literal(args[1])? as u32;
                Ok(StatementKind::SetDiscriminant {
                    place: Box::new(place),
                    variant_index: VariantIdx::from_u32(var),
                })
            },
            ExprKind::Assign { lhs, rhs } => {
                let lhs = self.parse_place(*lhs)?;
                let rhs = self.parse_rvalue(*rhs)?;
                Ok(StatementKind::Assign(Box::new((lhs, rhs))))
            },
        )
    }

    pub(crate) fn parse_terminator(&self, expr_id: ExprId) -> PResult<TerminatorKind<'tcx>> {
        parse_by_kind!(self, expr_id, expr, "terminator",
            @call(mir_return, _args) => {
                Ok(TerminatorKind::Return)
            },
            @call(mir_goto, args) => {
                Ok(TerminatorKind::Goto { target: self.parse_block(args[0])? } )
            },
            @call(mir_unreachable, _args) => {
                Ok(TerminatorKind::Unreachable)
            },
            @call(mir_unwind_resume, _args) => {
                Ok(TerminatorKind::UnwindResume)
            },
            @call(mir_unwind_terminate, args) => {
                Ok(TerminatorKind::UnwindTerminate(self.parse_unwind_terminate_reason(args[0])?))
            },
            @call(mir_drop, args) => {
                Ok(TerminatorKind::Drop {
                    place: self.parse_place(args[0])?,
                    target: self.parse_return_to(args[1])?,
                    unwind: self.parse_unwind_action(args[2])?,
                    replace: false,
                    drop: None,
                    async_fut: None,
                })
            },
            @call(mir_call, args) => {
                self.parse_call(args)
            },
            @call(mir_tail_call, args) => {
                self.parse_tail_call(args)
            },
            ExprKind::Match { scrutinee, arms, .. } => {
                let discr = self.parse_operand(*scrutinee)?;
                self.parse_match(arms, expr.span).map(|t| TerminatorKind::SwitchInt { discr, targets: t })
            },
        )
    }

    fn parse_unwind_terminate_reason(&self, expr_id: ExprId) -> PResult<UnwindTerminateReason> {
        parse_by_kind!(self, expr_id, _, "unwind terminate reason",
            @variant(mir_unwind_terminate_reason, Abi) => {
                Ok(UnwindTerminateReason::Abi)
            },
            @variant(mir_unwind_terminate_reason, InCleanup) => {
                Ok(UnwindTerminateReason::InCleanup)
            },
        )
    }

    fn parse_unwind_action(&self, expr_id: ExprId) -> PResult<UnwindAction> {
        parse_by_kind!(self, expr_id, _, "unwind action",
            @call(mir_unwind_continue, _args) => {
                Ok(UnwindAction::Continue)
            },
            @call(mir_unwind_unreachable, _args) => {
                Ok(UnwindAction::Unreachable)
            },
            @call(mir_unwind_terminate, args) => {
                Ok(UnwindAction::Terminate(self.parse_unwind_terminate_reason(args[0])?))
            },
            @call(mir_unwind_cleanup, args) => {
                Ok(UnwindAction::Cleanup(self.parse_block(args[0])?))
            },
        )
    }

    fn parse_return_to(&self, expr_id: ExprId) -> PResult<BasicBlock> {
        parse_by_kind!(self, expr_id, _, "return block",
            @call(mir_return_to, args) => {
                self.parse_block(args[0])
            },
        )
    }

    fn parse_match(&self, arms: &[ArmId], span: Span) -> PResult<SwitchTargets> {
        let Some((otherwise, rest)) = arms.split_last() else {
            return Err(ParseError {
                span,
                item_description: "no arms".to_string(),
                expected: "at least one arm".to_string(),
            });
        };

        let otherwise = &self.thir[*otherwise];
        let PatKind::Wild = otherwise.pattern.kind else {
            return Err(ParseError {
                span: otherwise.span,
                item_description: format!("{:?}", otherwise.pattern.kind),
                expected: "wildcard pattern".to_string(),
            });
        };
        let otherwise = self.parse_block(otherwise.body)?;

        let mut values = Vec::new();
        let mut targets = Vec::new();
        for arm in rest {
            let arm = &self.thir[*arm];
            let value = match arm.pattern.kind {
                PatKind::Constant { value } => value,
                PatKind::ExpandedConstant { ref subpattern, def_id: _ }
                    if let PatKind::Constant { value } = subpattern.kind =>
                {
                    value
                }
                _ => {
                    return Err(ParseError {
                        span: arm.pattern.span,
                        item_description: format!("{:?}", arm.pattern.kind),
                        expected: "constant pattern".to_string(),
                    });
                }
            };
            values.push(value.eval_bits(self.tcx, self.typing_env));
            targets.push(self.parse_block(arm.body)?);
        }

        Ok(SwitchTargets::new(values.into_iter().zip(targets), otherwise))
    }

    fn parse_call(&self, args: &[ExprId]) -> PResult<TerminatorKind<'tcx>> {
        let (destination, call) = parse_by_kind!(self, args[0], _, "function call",
            ExprKind::Assign { lhs, rhs } => (*lhs, *rhs),
        );
        let destination = self.parse_place(destination)?;
        let target = self.parse_return_to(args[1])?;
        let unwind = self.parse_unwind_action(args[2])?;

        parse_by_kind!(self, call, _, "function call",
            ExprKind::Call { fun, args, from_hir_call, fn_span, .. } => {
                let fun = self.parse_operand(*fun)?;
                let args = args
                    .iter()
                    .map(|arg|
                        Ok(Spanned { node: self.parse_operand(*arg)?, span: self.thir.exprs[*arg].span  } )
                    )
                    .collect::<PResult<Box<[_]>>>()?;
                Ok(TerminatorKind::Call {
                    func: fun,
                    args,
                    destination,
                    target: Some(target),
                    unwind,
                    call_source: if *from_hir_call { CallSource::Normal } else {
                        CallSource::OverloadedOperator
                    },
                    fn_span: *fn_span,
                })
            },
        )
    }

    fn parse_tail_call(&self, args: &[ExprId]) -> PResult<TerminatorKind<'tcx>> {
        parse_by_kind!(self, args[0], _, "tail call",
            ExprKind::Call { fun, args, fn_span, .. } => {
                let fun = self.parse_operand(*fun)?;
                let args = args
                    .iter()
                    .map(|arg|
                        Ok(Spanned { node: self.parse_operand(*arg)?, span: self.thir.exprs[*arg].span  } )
                    )
                    .collect::<PResult<Box<[_]>>>()?;
                Ok(TerminatorKind::TailCall {
                    func: fun,
                    args,
                    fn_span: *fn_span,
                })
            },
        )
    }

    fn parse_rvalue(&self, expr_id: ExprId) -> PResult<Rvalue<'tcx>> {
        parse_by_kind!(self, expr_id, expr, "rvalue",
            @call(mir_discriminant, args) => self.parse_place(args[0]).map(Rvalue::Discriminant),
            @call(mir_cast_transmute, args) => {
                let source = self.parse_operand(args[0])?;
                Ok(Rvalue::Cast(CastKind::Transmute, source, expr.ty))
            },
            @call(mir_cast_ptr_to_ptr, args) => {
                let source = self.parse_operand(args[0])?;
                Ok(Rvalue::Cast(CastKind::PtrToPtr, source, expr.ty))
            },
            @call(mir_checked, args) => {
                parse_by_kind!(self, args[0], _, "binary op",
                    ExprKind::Binary { op, lhs, rhs } => {
                        if let Some(op_with_overflow) = op.wrapping_to_overflowing() {
                            Ok(Rvalue::BinaryOp(
                                op_with_overflow, Box::new((self.parse_operand(*lhs)?, self.parse_operand(*rhs)?))
                            ))
                        } else {
                            Err(self.expr_error(expr_id, "No WithOverflow form of this operator"))
                        }
                    },
                )
            },
            @call(mir_offset, args) => {
                let ptr = self.parse_operand(args[0])?;
                let offset = self.parse_operand(args[1])?;
                Ok(Rvalue::BinaryOp(BinOp::Offset, Box::new((ptr, offset))))
            },
            @call(mir_len, args) => Ok(Rvalue::Len(self.parse_place(args[0])?)),
            @call(mir_ptr_metadata, args) => Ok(Rvalue::UnaryOp(UnOp::PtrMetadata, self.parse_operand(args[0])?)),
            @call(mir_copy_for_deref, args) => Ok(Rvalue::CopyForDeref(self.parse_place(args[0])?)),
            ExprKind::Borrow { borrow_kind, arg } => Ok(
                Rvalue::Ref(self.tcx.lifetimes.re_erased, *borrow_kind, self.parse_place(*arg)?)
            ),
            ExprKind::RawBorrow { mutability, arg } => Ok(
                Rvalue::RawPtr((*mutability).into(), self.parse_place(*arg)?)
            ),
            ExprKind::Binary { op, lhs, rhs } =>  Ok(
                Rvalue::BinaryOp(*op, Box::new((self.parse_operand(*lhs)?, self.parse_operand(*rhs)?)))
            ),
            ExprKind::Unary { op, arg } => Ok(
                Rvalue::UnaryOp(*op, self.parse_operand(*arg)?)
            ),
            ExprKind::Repeat { value, count } => Ok(
                Rvalue::Repeat(self.parse_operand(*value)?, *count)
            ),
            ExprKind::Cast { source } => {
                let source = self.parse_operand(*source)?;
                let source_ty = source.ty(self.body.local_decls(), self.tcx);
                let cast_kind = mir_cast_kind(source_ty, expr.ty);
                Ok(Rvalue::Cast(cast_kind, source, expr.ty))
            },
            ExprKind::Tuple { fields } => Ok(
                Rvalue::Aggregate(
                    Box::new(AggregateKind::Tuple),
                    fields.iter().map(|e| self.parse_operand(*e)).collect::<Result<_, _>>()?
                )
            ),
            ExprKind::Array { fields } => {
                let elem_ty = expr.ty.builtin_index().expect("ty must be an array");
                Ok(Rvalue::Aggregate(
                    Box::new(AggregateKind::Array(elem_ty)),
                    fields.iter().map(|e| self.parse_operand(*e)).collect::<Result<_, _>>()?
                ))
            },
            ExprKind::Adt(box AdtExpr { adt_def, variant_index, args, fields, .. }) => {
                let is_union = adt_def.is_union();
                let active_field_index = is_union.then(|| fields[0].name);

                Ok(Rvalue::Aggregate(
                    Box::new(AggregateKind::Adt(adt_def.did(), *variant_index, args, None, active_field_index)),
                    fields.iter().map(|f| self.parse_operand(f.expr)).collect::<Result<_, _>>()?
                ))
            },
            _ => self.parse_operand(expr_id).map(Rvalue::Use),
        )
    }

    pub(crate) fn parse_operand(&self, expr_id: ExprId) -> PResult<Operand<'tcx>> {
        parse_by_kind!(self, expr_id, expr, "operand",
            @call(mir_move, args) => self.parse_place(args[0]).map(Operand::Move),
            @call(mir_static, args) => self.parse_static(args[0]),
            @call(mir_static_mut, args) => self.parse_static(args[0]),
            ExprKind::Literal { .. }
            | ExprKind::NamedConst { .. }
            | ExprKind::NonHirLiteral { .. }
            | ExprKind::ZstLiteral { .. }
            | ExprKind::ConstParam { .. }
            | ExprKind::ConstBlock { .. } => {
                Ok(Operand::Constant(Box::new(
                    as_constant_inner(expr, |_| None, self.tcx)
                )))
            },
            _ => self.parse_place(expr_id).map(Operand::Copy),
        )
    }

    fn parse_place(&self, expr_id: ExprId) -> PResult<Place<'tcx>> {
        self.parse_place_inner(expr_id).map(|(x, _)| x)
    }

    fn parse_place_inner(&self, expr_id: ExprId) -> PResult<(Place<'tcx>, PlaceTy<'tcx>)> {
        let (parent, proj) = parse_by_kind!(self, expr_id, expr, "place",
            @call(mir_field, args) => {
                let (parent, place_ty) = self.parse_place_inner(args[0])?;
                let field = FieldIdx::from_u32(self.parse_integer_literal(args[1])? as u32);
                let field_ty = PlaceTy::field_ty(self.tcx, place_ty.ty, place_ty.variant_index, field);
                let proj = PlaceElem::Field(field, field_ty);
                let place = parent.project_deeper(&[proj], self.tcx);
                return Ok((place, PlaceTy::from_ty(field_ty)));
            },
            @call(mir_variant, args) => {
                (args[0], PlaceElem::Downcast(
                    None,
                    VariantIdx::from_u32(self.parse_integer_literal(args[1])? as u32)
                ))
            },
            ExprKind::Deref { arg } => {
                parse_by_kind!(self, *arg, _, "does not matter",
                    @call(mir_make_place, args) => return self.parse_place_inner(args[0]),
                    _ => (*arg, PlaceElem::Deref),
                )
            },
            ExprKind::Index { lhs, index } => (*lhs, PlaceElem::Index(self.parse_local(*index)?)),
            ExprKind::Field { lhs, name: field, .. } => (*lhs, PlaceElem::Field(*field, expr.ty)),
            _ => {
                let place = self.parse_local(expr_id).map(Place::from)?;
                return Ok((place, PlaceTy::from_ty(expr.ty)))
            },
        );
        let (parent, ty) = self.parse_place_inner(parent)?;
        let place = parent.project_deeper(&[proj], self.tcx);
        let ty = ty.projection_ty(self.tcx, proj);
        Ok((place, ty))
    }

    fn parse_local(&self, expr_id: ExprId) -> PResult<Local> {
        parse_by_kind!(self, expr_id, _, "local",
            ExprKind::VarRef { id } => Ok(self.local_map[id]),
        )
    }

    fn parse_block(&self, expr_id: ExprId) -> PResult<BasicBlock> {
        parse_by_kind!(self, expr_id, _, "basic block",
            ExprKind::VarRef { id } => Ok(self.block_map[id]),
        )
    }

    fn parse_static(&self, expr_id: ExprId) -> PResult<Operand<'tcx>> {
        let expr_id = parse_by_kind!(self, expr_id, _, "static",
            ExprKind::Deref { arg } => *arg,
        );

        parse_by_kind!(self, expr_id, expr, "static",
            ExprKind::StaticRef { alloc_id, ty, .. } => {
                let const_val =
                    ConstValue::Scalar(Scalar::from_pointer((*alloc_id).into(), &self.tcx));
                let const_ = Const::Val(const_val, *ty);

                Ok(Operand::Constant(Box::new(ConstOperand {
                    span: expr.span,
                    user_ty: None,
                    const_
                })))
            },
        )
    }

    fn parse_integer_literal(&self, expr_id: ExprId) -> PResult<u128> {
        parse_by_kind!(self, expr_id, expr, "constant",
            ExprKind::Literal { .. }
            | ExprKind::NamedConst { .. }
            | ExprKind::NonHirLiteral { .. }
            | ExprKind::ConstBlock { .. } => Ok({
                let value = as_constant_inner(expr, |_| None, self.tcx);
                value.const_.eval_bits(self.tcx, self.typing_env)
            }),
        )
    }
}
