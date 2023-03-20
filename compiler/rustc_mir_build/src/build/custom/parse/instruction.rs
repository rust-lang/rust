use rustc_middle::mir::interpret::{ConstValue, Scalar};
use rustc_middle::mir::tcx::PlaceTy;
use rustc_middle::ty::cast::mir_cast_kind;
use rustc_middle::{mir::*, thir::*, ty};
use rustc_span::Span;
use rustc_target::abi::VariantIdx;

use crate::build::custom::ParseError;
use crate::build::expr::as_constant::as_constant_inner;

use super::{parse_by_kind, PResult, ParseCtxt};

impl<'tcx, 'body> ParseCtxt<'tcx, 'body> {
    pub fn parse_statement(&self, expr_id: ExprId) -> PResult<StatementKind<'tcx>> {
        parse_by_kind!(self, expr_id, _, "statement",
            @call("mir_storage_live", args) => {
                Ok(StatementKind::StorageLive(self.parse_local(args[0])?))
            },
            @call("mir_storage_dead", args) => {
                Ok(StatementKind::StorageDead(self.parse_local(args[0])?))
            },
            @call("mir_deinit", args) => {
                Ok(StatementKind::Deinit(Box::new(self.parse_place(args[0])?)))
            },
            @call("mir_retag", args) => {
                Ok(StatementKind::Retag(RetagKind::Default, Box::new(self.parse_place(args[0])?)))
            },
            @call("mir_set_discriminant", args) => {
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

    pub fn parse_terminator(&self, expr_id: ExprId) -> PResult<TerminatorKind<'tcx>> {
        parse_by_kind!(self, expr_id, expr, "terminator",
            @call("mir_return", _args) => {
                Ok(TerminatorKind::Return)
            },
            @call("mir_goto", args) => {
                Ok(TerminatorKind::Goto { target: self.parse_block(args[0])? } )
            },
            @call("mir_unreachable", _args) => {
                Ok(TerminatorKind::Unreachable)
            },
            @call("mir_drop", args) => {
                Ok(TerminatorKind::Drop {
                    place: self.parse_place(args[0])?,
                    target: self.parse_block(args[1])?,
                    unwind: None,
                })
            },
            @call("mir_call", args) => {
                let destination = self.parse_place(args[0])?;
                let target = self.parse_block(args[1])?;
                self.parse_call(args[2], destination, target)
            },
            ExprKind::Match { scrutinee, arms } => {
                let discr = self.parse_operand(*scrutinee)?;
                self.parse_match(arms, expr.span).map(|t| TerminatorKind::SwitchInt { discr, targets: t })
            },
        )
    }

    fn parse_match(&self, arms: &[ArmId], span: Span) -> PResult<SwitchTargets> {
        let Some((otherwise, rest)) = arms.split_last() else {
            return Err(ParseError {
                span,
                item_description: "no arms".to_string(),
                expected: "at least one arm".to_string(),
            })
        };

        let otherwise = &self.thir[*otherwise];
        let PatKind::Wild = otherwise.pattern.kind else {
            return Err(ParseError {
                span: otherwise.span,
                item_description: format!("{:?}", otherwise.pattern.kind),
                expected: "wildcard pattern".to_string(),
            })
        };
        let otherwise = self.parse_block(otherwise.body)?;

        let mut values = Vec::new();
        let mut targets = Vec::new();
        for arm in rest {
            let arm = &self.thir[*arm];
            let PatKind::Constant { value } = arm.pattern.kind else {
                return Err(ParseError {
                    span: arm.pattern.span,
                    item_description: format!("{:?}", arm.pattern.kind),
                    expected: "constant pattern".to_string(),
                })
            };
            values.push(value.eval_bits(self.tcx, self.param_env, arm.pattern.ty));
            targets.push(self.parse_block(arm.body)?);
        }

        Ok(SwitchTargets::new(values.into_iter().zip(targets), otherwise))
    }

    fn parse_call(
        &self,
        expr_id: ExprId,
        destination: Place<'tcx>,
        target: BasicBlock,
    ) -> PResult<TerminatorKind<'tcx>> {
        parse_by_kind!(self, expr_id, _, "function call",
            ExprKind::Call { fun, args, from_hir_call, fn_span, .. } => {
                let fun = self.parse_operand(*fun)?;
                let args = args
                    .iter()
                    .map(|arg| self.parse_operand(*arg))
                    .collect::<PResult<Vec<_>>>()?;
                Ok(TerminatorKind::Call {
                    func: fun,
                    args,
                    destination,
                    target: Some(target),
                    cleanup: None,
                    from_hir_call: *from_hir_call,
                    fn_span: *fn_span,
                })
            },
        )
    }

    fn parse_rvalue(&self, expr_id: ExprId) -> PResult<Rvalue<'tcx>> {
        parse_by_kind!(self, expr_id, expr, "rvalue",
            @call("mir_discriminant", args) => self.parse_place(args[0]).map(Rvalue::Discriminant),
            @call("mir_checked", args) => {
                parse_by_kind!(self, args[0], _, "binary op",
                    ExprKind::Binary { op, lhs, rhs } => Ok(Rvalue::CheckedBinaryOp(
                        *op, Box::new((self.parse_operand(*lhs)?, self.parse_operand(*rhs)?))
                    )),
                )
            },
            @call("mir_len", args) => Ok(Rvalue::Len(self.parse_place(args[0])?)),
            ExprKind::Borrow { borrow_kind, arg } => Ok(
                Rvalue::Ref(self.tcx.lifetimes.re_erased, *borrow_kind, self.parse_place(*arg)?)
            ),
            ExprKind::AddressOf { mutability, arg } => Ok(
                Rvalue::AddressOf(*mutability, self.parse_place(*arg)?)
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
            ExprKind::Adt(box AdtExpr{ adt_def, variant_index, substs, fields, .. }) => {
                let is_union = adt_def.is_union();
                let active_field_index = is_union.then(|| fields[0].name.index());

                Ok(Rvalue::Aggregate(
                    Box::new(AggregateKind::Adt(adt_def.did(), *variant_index, substs, None, active_field_index)),
                    fields.iter().map(|f| self.parse_operand(f.expr)).collect::<Result<_, _>>()?
                ))
            },
            _ => self.parse_operand(expr_id).map(Rvalue::Use),
        )
    }

    fn parse_operand(&self, expr_id: ExprId) -> PResult<Operand<'tcx>> {
        parse_by_kind!(self, expr_id, expr, "operand",
            @call("mir_move", args) => self.parse_place(args[0]).map(Operand::Move),
            @call("mir_static", args) => self.parse_static(args[0]),
            @call("mir_static_mut", args) => self.parse_static(args[0]),
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
            @call("mir_field", args) => {
                let (parent, ty) = self.parse_place_inner(args[0])?;
                let field = Field::from_u32(self.parse_integer_literal(args[1])? as u32);
                let field_ty = ty.field_ty(self.tcx, field);
                let proj = PlaceElem::Field(field, field_ty);
                let place = parent.project_deeper(&[proj], self.tcx);
                return Ok((place, PlaceTy::from_ty(field_ty)));
            },
            @call("mir_variant", args) => {
                (args[0], PlaceElem::Downcast(
                    None,
                    VariantIdx::from_u32(self.parse_integer_literal(args[1])? as u32)
                ))
            },
            ExprKind::Deref { arg } => {
                parse_by_kind!(self, *arg, _, "does not matter",
                    @call("mir_make_place", args) => return self.parse_place_inner(args[0]),
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
                let literal = ConstantKind::Val(const_val, *ty);

                Ok(Operand::Constant(Box::new(Constant {
                    span: expr.span,
                    user_ty: None,
                    literal
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
                value.literal.eval_bits(self.tcx, self.param_env, value.ty())
            }),
        )
    }
}
