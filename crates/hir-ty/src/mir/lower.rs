//! This module generates a polymorphic MIR from a hir body

use std::{iter, mem, sync::Arc};

use chalk_ir::{BoundVar, ConstData, DebruijnIndex, TyKind};
use hir_def::{
    body::Body,
    expr::{
        Array, BindingAnnotation, ExprId, LabelId, Literal, MatchArm, Pat, PatId, RecordLitField,
    },
    layout::LayoutError,
    resolver::{resolver_for_expr, ResolveValueResult, ValueNs},
    DefWithBodyId, EnumVariantId, HasModule,
};
use la_arena::ArenaMap;

use crate::{
    consteval::ConstEvalError, db::HirDatabase, layout::layout_of_ty, mapping::ToChalk,
    utils::generics, Adjust, AutoBorrow, CallableDefId, TyBuilder, TyExt,
};

use super::*;

#[derive(Debug, Clone, Copy)]
struct LoopBlocks {
    begin: BasicBlockId,
    end: BasicBlockId,
}

struct MirLowerCtx<'a> {
    result: MirBody,
    owner: DefWithBodyId,
    binding_locals: ArenaMap<PatId, LocalId>,
    current_loop_blocks: Option<LoopBlocks>,
    discr_temp: Option<Place>,
    db: &'a dyn HirDatabase,
    body: &'a Body,
    infer: &'a InferenceResult,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MirLowerError {
    ConstEvalError(Box<ConstEvalError>),
    LayoutError(LayoutError),
    IncompleteExpr,
    UnresolvedName,
    MissingFunctionDefinition,
    TypeError(&'static str),
    NotSupported(String),
    ContinueWithoutLoop,
    BreakWithoutLoop,
    Loop,
}

macro_rules! not_supported {
    ($x: expr) => {
        return Err(MirLowerError::NotSupported(format!($x)))
    };
}

impl From<ConstEvalError> for MirLowerError {
    fn from(value: ConstEvalError) -> Self {
        match value {
            ConstEvalError::MirLowerError(e) => e,
            _ => MirLowerError::ConstEvalError(Box::new(value)),
        }
    }
}

impl From<LayoutError> for MirLowerError {
    fn from(value: LayoutError) -> Self {
        MirLowerError::LayoutError(value)
    }
}

type Result<T> = std::result::Result<T, MirLowerError>;

impl MirLowerCtx<'_> {
    fn temp(&mut self, ty: Ty) -> Result<LocalId> {
        if matches!(ty.kind(Interner), TyKind::Slice(_) | TyKind::Dyn(_)) {
            not_supported!("unsized temporaries");
        }
        Ok(self.result.locals.alloc(Local { mutability: Mutability::Not, ty }))
    }

    fn lower_expr_as_place(&self, expr_id: ExprId) -> Option<Place> {
        let adjustments = self.infer.expr_adjustments.get(&expr_id);
        let mut r = self.lower_expr_as_place_without_adjust(expr_id)?;
        for adjustment in adjustments.iter().flat_map(|x| x.iter()) {
            match adjustment.kind {
                Adjust::NeverToAny => return Some(r),
                Adjust::Deref(None) => {
                    r.projection.push(ProjectionElem::Deref);
                }
                Adjust::Deref(Some(_)) => return None,
                Adjust::Borrow(_) => return None,
                Adjust::Pointer(_) => return None,
            }
        }
        Some(r)
    }

    fn lower_expr_as_place_without_adjust(&self, expr_id: ExprId) -> Option<Place> {
        match &self.body.exprs[expr_id] {
            Expr::Path(p) => {
                let resolver = resolver_for_expr(self.db.upcast(), self.owner, expr_id);
                let pr = resolver.resolve_path_in_value_ns(self.db.upcast(), p.mod_path())?;
                let pr = match pr {
                    ResolveValueResult::ValueNs(v) => v,
                    ResolveValueResult::Partial(..) => return None,
                };
                match pr {
                    ValueNs::LocalBinding(pat_id) => Some(self.binding_locals[pat_id].into()),
                    _ => None,
                }
            }
            Expr::UnaryOp { expr, op } => match op {
                hir_def::expr::UnaryOp::Deref => {
                    let mut r = self.lower_expr_as_place(*expr)?;
                    r.projection.push(ProjectionElem::Deref);
                    Some(r)
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn lower_expr_to_some_operand(
        &mut self,
        expr_id: ExprId,
        current: BasicBlockId,
    ) -> Result<(Operand, BasicBlockId)> {
        if !self.has_adjustments(expr_id) {
            match &self.body.exprs[expr_id] {
                Expr::Literal(l) => {
                    let ty = self.expr_ty(expr_id);
                    return Ok((self.lower_literal_to_operand(ty, l)?, current));
                }
                _ => (),
            }
        }
        let (p, current) = self.lower_expr_to_some_place(expr_id, current)?;
        Ok((Operand::Copy(p), current))
    }

    fn lower_expr_to_some_place(
        &mut self,
        expr_id: ExprId,
        prev_block: BasicBlockId,
    ) -> Result<(Place, BasicBlockId)> {
        if let Some(p) = self.lower_expr_as_place(expr_id) {
            return Ok((p, prev_block));
        }
        let mut ty = self.expr_ty(expr_id);
        if let Some(x) = self.infer.expr_adjustments.get(&expr_id) {
            if let Some(x) = x.last() {
                ty = x.target.clone();
            }
        }
        let place = self.temp(ty)?;
        Ok((place.into(), self.lower_expr_to_place(expr_id, place.into(), prev_block)?))
    }

    fn lower_expr_to_place(
        &mut self,
        expr_id: ExprId,
        place: Place,
        prev_block: BasicBlockId,
    ) -> Result<BasicBlockId> {
        if let Some(x) = self.infer.expr_adjustments.get(&expr_id) {
            if x.len() > 0 {
                let tmp = self.temp(self.expr_ty(expr_id))?;
                let current =
                    self.lower_expr_to_place_without_adjust(expr_id, tmp.into(), prev_block)?;
                let mut r = Place::from(tmp);
                for adjustment in x {
                    match &adjustment.kind {
                        Adjust::NeverToAny => (),
                        Adjust::Deref(None) => {
                            r.projection.push(ProjectionElem::Deref);
                        }
                        Adjust::Deref(Some(_)) => not_supported!("overloaded dereference"),
                        Adjust::Borrow(AutoBorrow::Ref(m) | AutoBorrow::RawPtr(m)) => {
                            let tmp = self.temp(adjustment.target.clone())?;
                            self.push_assignment(
                                current,
                                tmp.into(),
                                Rvalue::Ref(BorrowKind::from_chalk(*m), r),
                            );
                            r = tmp.into();
                        }
                        Adjust::Pointer(cast) => {
                            let target = &adjustment.target;
                            let tmp = self.temp(target.clone())?;
                            self.push_assignment(
                                current,
                                tmp.into(),
                                Rvalue::Cast(
                                    CastKind::Pointer(cast.clone()),
                                    Operand::Copy(r).into(),
                                    target.clone(),
                                ),
                            );
                            r = tmp.into();
                        }
                    }
                }
                self.push_assignment(current, place, Operand::Copy(r).into());
                return Ok(current);
            }
        }
        self.lower_expr_to_place_without_adjust(expr_id, place, prev_block)
    }

    fn lower_expr_to_place_without_adjust(
        &mut self,
        expr_id: ExprId,
        place: Place,
        mut current: BasicBlockId,
    ) -> Result<BasicBlockId> {
        match &self.body.exprs[expr_id] {
            Expr::Missing => Err(MirLowerError::IncompleteExpr),
            Expr::Path(p) => {
                let resolver = resolver_for_expr(self.db.upcast(), self.owner, expr_id);
                let pr = resolver
                    .resolve_path_in_value_ns(self.db.upcast(), p.mod_path())
                    .ok_or(MirLowerError::UnresolvedName)?;
                let pr = match pr {
                    ResolveValueResult::ValueNs(v) => v,
                    ResolveValueResult::Partial(..) => {
                        return match self
                            .infer
                            .assoc_resolutions_for_expr(expr_id)
                            .ok_or(MirLowerError::UnresolvedName)?
                            .0
                            //.ok_or(ConstEvalError::SemanticError("unresolved assoc item"))?
                        {
                            hir_def::AssocItemId::ConstId(c) => self.lower_const(c, current, place),
                            _ => return Err(MirLowerError::UnresolvedName),
                        };
                    }
                };
                match pr {
                    ValueNs::LocalBinding(pat_id) => {
                        self.push_assignment(
                            current,
                            place,
                            Operand::Copy(self.binding_locals[pat_id].into()).into(),
                        );
                        Ok(current)
                    }
                    ValueNs::ConstId(const_id) => self.lower_const(const_id, current, place),
                    ValueNs::EnumVariantId(variant_id) => {
                        let ty = self.infer.type_of_expr[expr_id].clone();
                        self.lower_enum_variant(variant_id, current, place, ty, vec![])
                    }
                    ValueNs::GenericParam(p) => {
                        let Some(def) = self.owner.as_generic_def_id() else {
                            not_supported!("owner without generic def id");
                        };
                        let gen = generics(self.db.upcast(), def);
                        let ty = self.expr_ty(expr_id);
                        self.push_assignment(
                            current,
                            place,
                            Operand::Constant(
                                ConstData {
                                    ty,
                                    value: chalk_ir::ConstValue::BoundVar(BoundVar::new(
                                        DebruijnIndex::INNERMOST,
                                        gen.param_idx(p.into()).ok_or(MirLowerError::TypeError(
                                            "fail to lower const generic param",
                                        ))?,
                                    )),
                                }
                                .intern(Interner),
                            )
                            .into(),
                        );
                        Ok(current)
                    }
                    ValueNs::StructId(_) => {
                        // It's probably a unit struct or a zero sized function, so no action is needed.
                        Ok(current)
                    }
                    x => {
                        not_supported!("unknown name {x:?} in value name space");
                    }
                }
            }
            Expr::If { condition, then_branch, else_branch } => {
                let (discr, current) = self.lower_expr_to_some_operand(*condition, current)?;
                let start_of_then = self.new_basic_block();
                let end = self.new_basic_block();
                let end_of_then =
                    self.lower_expr_to_place(*then_branch, place.clone(), start_of_then)?;
                self.set_goto(end_of_then, end);
                let mut start_of_else = end;
                if let Some(else_branch) = else_branch {
                    start_of_else = self.new_basic_block();
                    let end_of_else =
                        self.lower_expr_to_place(*else_branch, place, start_of_else)?;
                    self.set_goto(end_of_else, end);
                }
                self.set_terminator(
                    current,
                    Terminator::SwitchInt {
                        discr,
                        targets: SwitchTargets::static_if(1, start_of_then, start_of_else),
                    },
                );
                Ok(end)
            }
            Expr::Let { pat, expr } => {
                let (cond_place, current) = self.lower_expr_to_some_place(*expr, current)?;
                let result = self.new_basic_block();
                let (then_target, else_target) = self.pattern_match(
                    current,
                    None,
                    cond_place,
                    self.expr_ty(*expr),
                    *pat,
                    BindingAnnotation::Unannotated,
                )?;
                self.write_bytes_to_place(then_target, place.clone(), vec![1], TyBuilder::bool())?;
                self.set_goto(then_target, result);
                if let Some(else_target) = else_target {
                    self.write_bytes_to_place(else_target, place, vec![0], TyBuilder::bool())?;
                    self.set_goto(else_target, result);
                }
                Ok(result)
            }
            Expr::Block { id: _, statements, tail, label } => {
                if label.is_some() {
                    not_supported!("block with label");
                }
                for statement in statements.iter() {
                    match statement {
                        hir_def::expr::Statement::Let {
                            pat,
                            initializer,
                            else_branch,
                            type_ref: _,
                        } => match initializer {
                            Some(expr_id) => {
                                let else_block;
                                let init_place;
                                (init_place, current) =
                                    self.lower_expr_to_some_place(*expr_id, current)?;
                                (current, else_block) = self.pattern_match(
                                    current,
                                    None,
                                    init_place,
                                    self.expr_ty(*expr_id),
                                    *pat,
                                    BindingAnnotation::Unannotated,
                                )?;
                                match (else_block, else_branch) {
                                    (None, _) => (),
                                    (Some(else_block), None) => {
                                        self.set_terminator(else_block, Terminator::Unreachable);
                                    }
                                    (Some(else_block), Some(else_branch)) => {
                                        let (_, b) = self
                                            .lower_expr_to_some_place(*else_branch, else_block)?;
                                        self.set_terminator(b, Terminator::Unreachable);
                                    }
                                }
                            }
                            None => continue,
                        },
                        hir_def::expr::Statement::Expr { expr, has_semi: _ } => {
                            let ty = self.expr_ty(*expr);
                            let temp = self.temp(ty)?;
                            current = self.lower_expr_to_place(*expr, temp.into(), current)?;
                        }
                    }
                }
                match tail {
                    Some(tail) => self.lower_expr_to_place(*tail, place, current),
                    None => Ok(current),
                }
            }
            Expr::Loop { body, label } => self.lower_loop(current, *label, |this, begin, _| {
                let (_, block) = this.lower_expr_to_some_place(*body, begin)?;
                this.set_goto(block, begin);
                Ok(())
            }),
            Expr::While { condition, body, label } => {
                self.lower_loop(current, *label, |this, begin, end| {
                    let (discr, to_switch) = this.lower_expr_to_some_operand(*condition, begin)?;
                    let after_cond = this.new_basic_block();
                    this.set_terminator(
                        to_switch,
                        Terminator::SwitchInt {
                            discr,
                            targets: SwitchTargets::static_if(1, after_cond, end),
                        },
                    );
                    let (_, block) = this.lower_expr_to_some_place(*body, after_cond)?;
                    this.set_goto(block, begin);
                    Ok(())
                })
            }
            Expr::For { .. } => not_supported!("for loop"),
            Expr::Call { callee, args, .. } => {
                let callee_ty = self.expr_ty(*callee);
                match &callee_ty.data(Interner).kind {
                    chalk_ir::TyKind::FnDef(..) => {
                        let func = Operand::from_bytes(vec![], callee_ty.clone());
                        self.lower_call(func, args.iter().copied(), place, current)
                    }
                    TyKind::Scalar(_)
                    | TyKind::Tuple(_, _)
                    | TyKind::Array(_, _)
                    | TyKind::Adt(_, _)
                    | TyKind::Str
                    | TyKind::Foreign(_)
                    | TyKind::Slice(_) => {
                        return Err(MirLowerError::TypeError("function call on data type"))
                    }
                    TyKind::Error => return Err(MirLowerError::MissingFunctionDefinition),
                    TyKind::AssociatedType(_, _)
                    | TyKind::Raw(_, _)
                    | TyKind::Ref(_, _, _)
                    | TyKind::OpaqueType(_, _)
                    | TyKind::Never
                    | TyKind::Closure(_, _)
                    | TyKind::Generator(_, _)
                    | TyKind::GeneratorWitness(_, _)
                    | TyKind::Placeholder(_)
                    | TyKind::Dyn(_)
                    | TyKind::Alias(_)
                    | TyKind::Function(_)
                    | TyKind::BoundVar(_)
                    | TyKind::InferenceVar(_, _) => not_supported!("dynamic function call"),
                }
            }
            Expr::MethodCall { receiver, args, .. } => {
                let (func_id, generic_args) =
                    self.infer.method_resolution(expr_id).ok_or(MirLowerError::UnresolvedName)?;
                let ty = chalk_ir::TyKind::FnDef(
                    CallableDefId::FunctionId(func_id).to_chalk(self.db),
                    generic_args,
                )
                .intern(Interner);
                let func = Operand::from_bytes(vec![], ty);
                self.lower_call(
                    func,
                    iter::once(*receiver).chain(args.iter().copied()),
                    place,
                    current,
                )
            }
            Expr::Match { expr, arms } => {
                let (cond_place, mut current) = self.lower_expr_to_some_place(*expr, current)?;
                let cond_ty = self.expr_ty(*expr);
                let end = self.new_basic_block();
                for MatchArm { pat, guard, expr } in arms.iter() {
                    if guard.is_some() {
                        not_supported!("pattern matching with guard");
                    }
                    let (then, otherwise) = self.pattern_match(
                        current,
                        None,
                        cond_place.clone(),
                        cond_ty.clone(),
                        *pat,
                        BindingAnnotation::Unannotated,
                    )?;
                    let block = self.lower_expr_to_place(*expr, place.clone(), then)?;
                    self.set_goto(block, end);
                    match otherwise {
                        Some(o) => current = o,
                        None => {
                            // The current pattern was irrefutable, so there is no need to generate code
                            // for the rest of patterns
                            break;
                        }
                    }
                }
                if self.is_unterminated(current) {
                    self.set_terminator(current, Terminator::Unreachable);
                }
                Ok(end)
            }
            Expr::Continue { label } => match label {
                Some(_) => not_supported!("continue with label"),
                None => {
                    let loop_data =
                        self.current_loop_blocks.ok_or(MirLowerError::ContinueWithoutLoop)?;
                    self.set_goto(current, loop_data.begin);
                    let otherwise = self.new_basic_block();
                    Ok(otherwise)
                }
            },
            Expr::Break { expr, label } => {
                if expr.is_some() {
                    not_supported!("break with value");
                }
                match label {
                    Some(_) => not_supported!("break with label"),
                    None => {
                        let loop_data =
                            self.current_loop_blocks.ok_or(MirLowerError::BreakWithoutLoop)?;
                        self.set_goto(current, loop_data.end);
                        Ok(self.new_basic_block())
                    }
                }
            }
            Expr::Return { expr } => {
                if let Some(expr) = expr {
                    current = self.lower_expr_to_place(*expr, return_slot().into(), current)?;
                }
                self.set_terminator(current, Terminator::Return);
                Ok(self.new_basic_block())
            }
            Expr::Yield { .. } => not_supported!("yield"),
            Expr::RecordLit { fields, .. } => {
                let variant_id = self
                    .infer
                    .variant_resolution_for_expr(expr_id)
                    .ok_or(MirLowerError::UnresolvedName)?;
                let subst = match self.expr_ty(expr_id).kind(Interner) {
                    TyKind::Adt(_, s) => s.clone(),
                    _ => not_supported!("Non ADT record literal"),
                };
                let variant_data = variant_id.variant_data(self.db.upcast());
                match variant_id {
                    VariantId::EnumVariantId(_) | VariantId::StructId(_) => {
                        let mut operands = vec![None; variant_data.fields().len()];
                        for RecordLitField { name, expr } in fields.iter() {
                            let field_id =
                                variant_data.field(name).ok_or(MirLowerError::UnresolvedName)?;
                            let op;
                            (op, current) = self.lower_expr_to_some_operand(*expr, current)?;
                            operands[u32::from(field_id.into_raw()) as usize] = Some(op);
                        }
                        self.push_assignment(
                            current,
                            place,
                            Rvalue::Aggregate(
                                AggregateKind::Adt(variant_id, subst),
                                operands.into_iter().map(|x| x).collect::<Option<_>>().ok_or(
                                    MirLowerError::TypeError("missing field in record literal"),
                                )?,
                            ),
                        );
                        Ok(current)
                    }
                    VariantId::UnionId(union_id) => {
                        let [RecordLitField { name, expr }] = fields.as_ref() else {
                            not_supported!("Union record literal with more than one field");
                        };
                        let local_id =
                            variant_data.field(name).ok_or(MirLowerError::UnresolvedName)?;
                        let mut place = place;
                        place
                            .projection
                            .push(PlaceElem::Field(FieldId { parent: union_id.into(), local_id }));
                        self.lower_expr_to_place(*expr, place, current)
                    }
                }
            }
            Expr::Field { expr, name } => {
                let (mut current_place, current) = self.lower_expr_to_some_place(*expr, current)?;
                if let TyKind::Tuple(..) = self.expr_ty(*expr).kind(Interner) {
                    let index = name
                        .as_tuple_index()
                        .ok_or(MirLowerError::TypeError("named field on tuple"))?;
                    current_place.projection.push(ProjectionElem::TupleField(index))
                } else {
                    let field = self
                        .infer
                        .field_resolution(expr_id)
                        .ok_or(MirLowerError::UnresolvedName)?;
                    current_place.projection.push(ProjectionElem::Field(field));
                }
                self.push_assignment(current, place, Operand::Copy(current_place).into());
                Ok(current)
            }
            Expr::Await { .. } => not_supported!("await"),
            Expr::Try { .. } => not_supported!("? operator"),
            Expr::Yeet { .. } => not_supported!("yeet"),
            Expr::TryBlock { .. } => not_supported!("try block"),
            Expr::Async { .. } => not_supported!("async block"),
            Expr::Const { .. } => not_supported!("anonymous const block"),
            Expr::Cast { expr, type_ref: _ } => {
                let (x, current) = self.lower_expr_to_some_operand(*expr, current)?;
                let source_ty = self.infer[*expr].clone();
                let target_ty = self.infer[expr_id].clone();
                self.push_assignment(
                    current,
                    place,
                    Rvalue::Cast(cast_kind(&source_ty, &target_ty)?, x, target_ty),
                );
                Ok(current)
            }
            Expr::Ref { expr, rawness: _, mutability } => {
                let p;
                (p, current) = self.lower_expr_to_some_place(*expr, current)?;
                let bk = BorrowKind::from_hir(*mutability);
                self.push_assignment(current, place, Rvalue::Ref(bk, p));
                Ok(current)
            }
            Expr::Box { .. } => not_supported!("box expression"),
            Expr::UnaryOp { expr, op } => match op {
                hir_def::expr::UnaryOp::Deref => {
                    let (mut tmp, current) = self.lower_expr_to_some_place(*expr, current)?;
                    tmp.projection.push(ProjectionElem::Deref);
                    self.push_assignment(current, place, Operand::Copy(tmp).into());
                    Ok(current)
                }
                hir_def::expr::UnaryOp::Not => {
                    let (op, current) = self.lower_expr_to_some_operand(*expr, current)?;
                    self.push_assignment(current, place, Rvalue::UnaryOp(UnOp::Not, op));
                    Ok(current)
                }
                hir_def::expr::UnaryOp::Neg => {
                    let (op, current) = self.lower_expr_to_some_operand(*expr, current)?;
                    self.push_assignment(current, place, Rvalue::UnaryOp(UnOp::Neg, op));
                    Ok(current)
                }
            },
            Expr::BinaryOp { lhs, rhs, op } => {
                let op = op.ok_or(MirLowerError::IncompleteExpr)?;
                if let hir_def::expr::BinaryOp::Assignment { op } = op {
                    if op.is_some() {
                        not_supported!("assignment with arith op (like +=)");
                    }
                    let Some(lhs_place) = self.lower_expr_as_place(*lhs) else {
                        not_supported!("assignment to complex place");
                    };
                    let rhs_op;
                    (rhs_op, current) = self.lower_expr_to_some_operand(*rhs, current)?;
                    self.push_assignment(current, lhs_place, rhs_op.into());
                    return Ok(current);
                }
                let lhs_op;
                (lhs_op, current) = self.lower_expr_to_some_operand(*lhs, current)?;
                let rhs_op;
                (rhs_op, current) = self.lower_expr_to_some_operand(*rhs, current)?;
                self.push_assignment(
                    current,
                    place,
                    Rvalue::CheckedBinaryOp(
                        match op {
                            hir_def::expr::BinaryOp::LogicOp(op) => match op {
                                hir_def::expr::LogicOp::And => BinOp::BitAnd, // FIXME: make these short circuit
                                hir_def::expr::LogicOp::Or => BinOp::BitOr,
                            },
                            hir_def::expr::BinaryOp::ArithOp(op) => BinOp::from(op),
                            hir_def::expr::BinaryOp::CmpOp(op) => BinOp::from(op),
                            hir_def::expr::BinaryOp::Assignment { .. } => unreachable!(), // handled above
                        },
                        lhs_op,
                        rhs_op,
                    ),
                );
                Ok(current)
            }
            Expr::Range { .. } => not_supported!("range"),
            Expr::Index { base, index } => {
                let mut p_base;
                (p_base, current) = self.lower_expr_to_some_place(*base, current)?;
                let l_index = self.temp(self.expr_ty(*index))?;
                current = self.lower_expr_to_place(*index, l_index.into(), current)?;
                p_base.projection.push(ProjectionElem::Index(l_index));
                self.push_assignment(current, place, Operand::Copy(p_base).into());
                Ok(current)
            }
            Expr::Closure { .. } => not_supported!("closure"),
            Expr::Tuple { exprs, is_assignee_expr: _ } => {
                let r = Rvalue::Aggregate(
                    AggregateKind::Tuple(self.expr_ty(expr_id)),
                    exprs
                        .iter()
                        .map(|x| {
                            let o;
                            (o, current) = self.lower_expr_to_some_operand(*x, current)?;
                            Ok(o)
                        })
                        .collect::<Result<_>>()?,
                );
                self.push_assignment(current, place, r);
                Ok(current)
            }
            Expr::Unsafe { body } => self.lower_expr_to_place(*body, place, current),
            Expr::Array(l) => match l {
                Array::ElementList { elements, .. } => {
                    let elem_ty = match &self.expr_ty(expr_id).data(Interner).kind {
                        TyKind::Array(ty, _) => ty.clone(),
                        _ => {
                            return Err(MirLowerError::TypeError(
                                "Array expression with non array type",
                            ))
                        }
                    };
                    let r = Rvalue::Aggregate(
                        AggregateKind::Array(elem_ty),
                        elements
                            .iter()
                            .map(|x| {
                                let o;
                                (o, current) = self.lower_expr_to_some_operand(*x, current)?;
                                Ok(o)
                            })
                            .collect::<Result<_>>()?,
                    );
                    self.push_assignment(current, place, r);
                    Ok(current)
                }
                Array::Repeat { .. } => not_supported!("array repeat"),
            },
            Expr::Literal(l) => {
                let ty = self.expr_ty(expr_id);
                let op = self.lower_literal_to_operand(ty, l)?;
                self.push_assignment(current, place, op.into());
                Ok(current)
            }
            Expr::Underscore => not_supported!("underscore"),
        }
    }

    fn lower_literal_to_operand(&mut self, ty: Ty, l: &Literal) -> Result<Operand> {
        let size = layout_of_ty(self.db, &ty, self.owner.module(self.db.upcast()).krate())?
            .size
            .bytes_usize();
        let bytes = match l {
            hir_def::expr::Literal::String(b) => {
                let b = b.as_bytes();
                let mut data = vec![];
                data.extend(0usize.to_le_bytes());
                data.extend(b.len().to_le_bytes());
                let mut mm = MemoryMap::default();
                mm.insert(0, b.to_vec());
                return Ok(Operand::from_concrete_const(data, mm, ty));
            }
            hir_def::expr::Literal::ByteString(b) => {
                let mut data = vec![];
                data.extend(0usize.to_le_bytes());
                data.extend(b.len().to_le_bytes());
                let mut mm = MemoryMap::default();
                mm.insert(0, b.to_vec());
                return Ok(Operand::from_concrete_const(data, mm, ty));
            }
            hir_def::expr::Literal::Char(c) => u32::from(*c).to_le_bytes().into(),
            hir_def::expr::Literal::Bool(b) => vec![*b as u8],
            hir_def::expr::Literal::Int(x, _) => x.to_le_bytes()[0..size].into(),
            hir_def::expr::Literal::Uint(x, _) => x.to_le_bytes()[0..size].into(),
            hir_def::expr::Literal::Float(f, _) => match size {
                8 => f.into_f64().to_le_bytes().into(),
                4 => f.into_f32().to_le_bytes().into(),
                _ => {
                    return Err(MirLowerError::TypeError("float with size other than 4 or 8 bytes"))
                }
            },
        };
        Ok(Operand::from_concrete_const(bytes, MemoryMap::default(), ty))
    }

    fn new_basic_block(&mut self) -> BasicBlockId {
        self.result.basic_blocks.alloc(BasicBlock::default())
    }

    fn lower_const(
        &mut self,
        const_id: hir_def::ConstId,
        prev_block: BasicBlockId,
        place: Place,
    ) -> Result<BasicBlockId> {
        let c = self.db.const_eval(const_id)?;
        self.write_const_to_place(c, prev_block, place)
    }

    fn write_const_to_place(
        &mut self,
        c: Const,
        prev_block: BasicBlockId,
        place: Place,
    ) -> Result<BasicBlockId> {
        self.push_assignment(prev_block, place, Operand::Constant(c).into());
        Ok(prev_block)
    }

    fn write_bytes_to_place(
        &mut self,
        prev_block: BasicBlockId,
        place: Place,
        cv: Vec<u8>,
        ty: Ty,
    ) -> Result<BasicBlockId> {
        self.push_assignment(prev_block, place, Operand::from_bytes(cv, ty).into());
        Ok(prev_block)
    }

    fn lower_enum_variant(
        &mut self,
        variant_id: EnumVariantId,
        prev_block: BasicBlockId,
        place: Place,
        ty: Ty,
        fields: Vec<Operand>,
    ) -> Result<BasicBlockId> {
        let subst = match ty.kind(Interner) {
            TyKind::Adt(_, subst) => subst.clone(),
            _ => not_supported!("Non ADT enum"),
        };
        self.push_assignment(
            prev_block,
            place,
            Rvalue::Aggregate(AggregateKind::Adt(variant_id.into(), subst), fields),
        );
        Ok(prev_block)
    }

    fn lower_call(
        &mut self,
        func: Operand,
        args: impl Iterator<Item = ExprId>,
        place: Place,
        mut current: BasicBlockId,
    ) -> Result<BasicBlockId> {
        let args = args
            .map(|arg| {
                let temp;
                (temp, current) = self.lower_expr_to_some_operand(arg, current)?;
                Ok(temp)
            })
            .collect::<Result<Vec<_>>>()?;
        let b = self.result.basic_blocks.alloc(BasicBlock {
            statements: vec![],
            terminator: None,
            is_cleanup: false,
        });
        self.set_terminator(
            current,
            Terminator::Call {
                func,
                args,
                destination: place,
                target: Some(b),
                cleanup: None,
                from_hir_call: true,
            },
        );
        Ok(b)
    }

    fn is_unterminated(&mut self, source: BasicBlockId) -> bool {
        self.result.basic_blocks[source].terminator.is_none()
    }

    fn set_terminator(&mut self, source: BasicBlockId, terminator: Terminator) {
        self.result.basic_blocks[source].terminator = Some(terminator);
    }

    fn set_goto(&mut self, source: BasicBlockId, target: BasicBlockId) {
        self.set_terminator(source, Terminator::Goto { target });
    }

    fn expr_ty(&self, e: ExprId) -> Ty {
        self.infer[e].clone()
    }

    fn push_assignment(&mut self, block: BasicBlockId, place: Place, rvalue: Rvalue) {
        self.result.basic_blocks[block].statements.push(Statement::Assign(place, rvalue));
    }

    /// It gets a `current` unterminated block, appends some statements and possibly a terminator to it to check if
    /// the pattern matches and write bindings, and returns two unterminated blocks, one for the matched path (which
    /// can be the `current` block) and one for the mismatched path. If the input pattern is irrefutable, the
    /// mismatched path block is `None`.
    ///
    /// By default, it will create a new block for mismatched path. If you already have one, you can provide it with
    /// `current_else` argument to save an unneccessary jump. If `current_else` isn't `None`, the result mismatched path
    /// wouldn't be `None` as well. Note that this function will add jumps to the beginning of the `current_else` block,
    /// so it should be an empty block.
    fn pattern_match(
        &mut self,
        mut current: BasicBlockId,
        mut current_else: Option<BasicBlockId>,
        mut cond_place: Place,
        mut cond_ty: Ty,
        pattern: PatId,
        mut binding_mode: BindingAnnotation,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        Ok(match &self.body.pats[pattern] {
            Pat::Missing => return Err(MirLowerError::IncompleteExpr),
            Pat::Wild => (current, current_else),
            Pat::Tuple { args, ellipsis } => {
                pattern_matching_dereference(&mut cond_ty, &mut binding_mode, &mut cond_place);
                let subst = match cond_ty.kind(Interner) {
                    TyKind::Tuple(_, s) => s,
                    _ => {
                        return Err(MirLowerError::TypeError(
                            "non tuple type matched with tuple pattern",
                        ))
                    }
                };
                self.pattern_match_tuple_like(
                    current,
                    current_else,
                    args.iter().enumerate().map(|(i, x)| {
                        (
                            PlaceElem::TupleField(i),
                            *x,
                            subst.at(Interner, i).assert_ty_ref(Interner).clone(),
                        )
                    }),
                    *ellipsis,
                    &cond_place,
                    binding_mode,
                )?
            }
            Pat::Or(_) => not_supported!("or pattern"),
            Pat::Record { .. } => not_supported!("record pattern"),
            Pat::Range { .. } => not_supported!("range pattern"),
            Pat::Slice { .. } => not_supported!("slice pattern"),
            Pat::Path(_) => not_supported!("path pattern"),
            Pat::Lit(l) => {
                let then_target = self.new_basic_block();
                let else_target = current_else.unwrap_or_else(|| self.new_basic_block());
                match &self.body.exprs[*l] {
                    Expr::Literal(l) => match l {
                        hir_def::expr::Literal::Int(x, _) => {
                            self.set_terminator(
                                current,
                                Terminator::SwitchInt {
                                    discr: Operand::Copy(cond_place),
                                    targets: SwitchTargets::static_if(
                                        *x as u128,
                                        then_target,
                                        else_target,
                                    ),
                                },
                            );
                        }
                        hir_def::expr::Literal::Uint(x, _) => {
                            self.set_terminator(
                                current,
                                Terminator::SwitchInt {
                                    discr: Operand::Copy(cond_place),
                                    targets: SwitchTargets::static_if(*x, then_target, else_target),
                                },
                            );
                        }
                        _ => not_supported!("non int path literal"),
                    },
                    _ => not_supported!("expression path literal"),
                }
                (then_target, Some(else_target))
            }
            Pat::Bind { mode, name: _, subpat } => {
                let target_place = self.binding_locals[pattern];
                if let Some(subpat) = subpat {
                    (current, current_else) = self.pattern_match(
                        current,
                        current_else,
                        cond_place.clone(),
                        cond_ty,
                        *subpat,
                        binding_mode,
                    )?
                }
                if matches!(mode, BindingAnnotation::Ref | BindingAnnotation::RefMut) {
                    binding_mode = *mode;
                }
                self.push_assignment(
                    current,
                    target_place.into(),
                    match binding_mode {
                        BindingAnnotation::Unannotated | BindingAnnotation::Mutable => {
                            Operand::Copy(cond_place).into()
                        }
                        BindingAnnotation::Ref => Rvalue::Ref(BorrowKind::Shared, cond_place),
                        BindingAnnotation::RefMut => Rvalue::Ref(
                            BorrowKind::Mut { allow_two_phase_borrow: false },
                            cond_place,
                        ),
                    },
                );
                (current, current_else)
            }
            Pat::TupleStruct { path: _, args, ellipsis } => {
                let Some(variant) = self.infer.variant_resolution_for_pat(pattern) else {
                    not_supported!("unresolved variant");
                };
                pattern_matching_dereference(&mut cond_ty, &mut binding_mode, &mut cond_place);
                let subst = match cond_ty.kind(Interner) {
                    TyKind::Adt(_, s) => s,
                    _ => {
                        return Err(MirLowerError::TypeError(
                            "non adt type matched with tuple struct",
                        ))
                    }
                };
                let fields_type = self.db.field_types(variant);
                match variant {
                    VariantId::EnumVariantId(v) => {
                        let e = self.db.const_eval_discriminant(v)? as u128;
                        let next = self.new_basic_block();
                        let tmp = self.discr_temp_place();
                        self.push_assignment(
                            current,
                            tmp.clone(),
                            Rvalue::Discriminant(cond_place.clone()),
                        );
                        let else_target = current_else.unwrap_or_else(|| self.new_basic_block());
                        self.set_terminator(
                            current,
                            Terminator::SwitchInt {
                                discr: Operand::Copy(tmp),
                                targets: SwitchTargets::static_if(e, next, else_target),
                            },
                        );
                        let enum_data = self.db.enum_data(v.parent);
                        let fields =
                            enum_data.variants[v.local_id].variant_data.fields().iter().map(
                                |(x, _)| {
                                    (
                                        PlaceElem::Field(FieldId { parent: v.into(), local_id: x }),
                                        fields_type[x].clone().substitute(Interner, subst),
                                    )
                                },
                            );
                        self.pattern_match_tuple_like(
                            next,
                            Some(else_target),
                            args.iter().zip(fields).map(|(x, y)| (y.0, *x, y.1)),
                            *ellipsis,
                            &cond_place,
                            binding_mode,
                        )?
                    }
                    VariantId::StructId(s) => {
                        let struct_data = self.db.struct_data(s);
                        let fields = struct_data.variant_data.fields().iter().map(|(x, _)| {
                            (
                                PlaceElem::Field(FieldId { parent: s.into(), local_id: x }),
                                fields_type[x].clone().substitute(Interner, subst),
                            )
                        });
                        self.pattern_match_tuple_like(
                            current,
                            current_else,
                            args.iter().zip(fields).map(|(x, y)| (y.0, *x, y.1)),
                            *ellipsis,
                            &cond_place,
                            binding_mode,
                        )?
                    }
                    VariantId::UnionId(_) => {
                        return Err(MirLowerError::TypeError("pattern matching on union"))
                    }
                }
            }
            Pat::Ref { .. } => not_supported!("& pattern"),
            Pat::Box { .. } => not_supported!("box pattern"),
            Pat::ConstBlock(_) => not_supported!("const block pattern"),
        })
    }

    fn pattern_match_tuple_like(
        &mut self,
        mut current: BasicBlockId,
        mut current_else: Option<BasicBlockId>,
        args: impl Iterator<Item = (PlaceElem, PatId, Ty)>,
        ellipsis: Option<usize>,
        cond_place: &Place,
        binding_mode: BindingAnnotation,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        if ellipsis.is_some() {
            not_supported!("tuple like pattern with ellipsis");
        }
        for (proj, arg, ty) in args {
            let mut cond_place = cond_place.clone();
            cond_place.projection.push(proj);
            (current, current_else) =
                self.pattern_match(current, current_else, cond_place, ty, arg, binding_mode)?;
        }
        Ok((current, current_else))
    }

    fn discr_temp_place(&mut self) -> Place {
        match &self.discr_temp {
            Some(x) => x.clone(),
            None => {
                let tmp: Place =
                    self.temp(TyBuilder::discr_ty()).expect("discr_ty is never unsized").into();
                self.discr_temp = Some(tmp.clone());
                tmp
            }
        }
    }

    fn lower_loop(
        &mut self,
        prev_block: BasicBlockId,
        label: Option<LabelId>,
        f: impl FnOnce(&mut MirLowerCtx<'_>, BasicBlockId, BasicBlockId) -> Result<()>,
    ) -> Result<BasicBlockId> {
        if label.is_some() {
            not_supported!("loop with label");
        }
        let begin = self.new_basic_block();
        let end = self.new_basic_block();
        let prev = mem::replace(&mut self.current_loop_blocks, Some(LoopBlocks { begin, end }));
        self.set_goto(prev_block, begin);
        f(self, begin, end)?;
        self.current_loop_blocks = prev;
        Ok(end)
    }

    fn has_adjustments(&self, expr_id: ExprId) -> bool {
        !self.infer.expr_adjustments.get(&expr_id).map(|x| x.is_empty()).unwrap_or(true)
    }
}

fn pattern_matching_dereference(
    cond_ty: &mut Ty,
    binding_mode: &mut BindingAnnotation,
    cond_place: &mut Place,
) {
    while let Some((ty, _, mu)) = cond_ty.as_reference() {
        if mu == Mutability::Mut && *binding_mode != BindingAnnotation::Ref {
            *binding_mode = BindingAnnotation::RefMut;
        } else {
            *binding_mode = BindingAnnotation::Ref;
        }
        *cond_ty = ty.clone();
        cond_place.projection.push(ProjectionElem::Deref);
    }
}

fn cast_kind(source_ty: &Ty, target_ty: &Ty) -> Result<CastKind> {
    Ok(match (source_ty.kind(Interner), target_ty.kind(Interner)) {
        (TyKind::Scalar(s), TyKind::Scalar(t)) => match (s, t) {
            (chalk_ir::Scalar::Float(_), chalk_ir::Scalar::Float(_)) => CastKind::FloatToFloat,
            (chalk_ir::Scalar::Float(_), _) => CastKind::FloatToInt,
            (_, chalk_ir::Scalar::Float(_)) => CastKind::IntToFloat,
            (_, _) => CastKind::IntToInt,
        },
        // Enum to int casts
        (TyKind::Scalar(_), TyKind::Adt(..)) | (TyKind::Adt(..), TyKind::Scalar(_)) => {
            CastKind::IntToInt
        }
        (a, b) => not_supported!("Unknown cast between {a:?} and {b:?}"),
    })
}

pub fn mir_body_query(db: &dyn HirDatabase, def: DefWithBodyId) -> Result<Arc<MirBody>> {
    let body = db.body(def);
    let infer = db.infer(def);
    Ok(Arc::new(lower_to_mir(db, def, &body, &infer, body.body_expr)?))
}

pub fn mir_body_recover(
    _db: &dyn HirDatabase,
    _cycle: &[String],
    _def: &DefWithBodyId,
) -> Result<Arc<MirBody>> {
    Err(MirLowerError::Loop)
}

pub fn lower_to_mir(
    db: &dyn HirDatabase,
    owner: DefWithBodyId,
    body: &Body,
    infer: &InferenceResult,
    // FIXME: root_expr should always be the body.body_expr, but since `X` in `[(); X]` doesn't have its own specific body yet, we
    // need to take this input explicitly.
    root_expr: ExprId,
) -> Result<MirBody> {
    let mut basic_blocks = Arena::new();
    let start_block =
        basic_blocks.alloc(BasicBlock { statements: vec![], terminator: None, is_cleanup: false });
    let mut locals = Arena::new();
    // 0 is return local
    locals.alloc(Local { mutability: Mutability::Mut, ty: infer[root_expr].clone() });
    let mut create_local_of_path = |p: PatId| {
        // FIXME: mutablity is broken
        locals.alloc(Local { mutability: Mutability::Not, ty: infer[p].clone() })
    };
    // 1 to param_len is for params
    let mut binding_locals: ArenaMap<PatId, LocalId> =
        body.params.iter().map(|&x| (x, create_local_of_path(x))).collect();
    // and then rest of bindings
    for (pat_id, _) in body.pats.iter() {
        if !binding_locals.contains_idx(pat_id) {
            binding_locals.insert(pat_id, create_local_of_path(pat_id));
        }
    }
    let mir = MirBody { basic_blocks, locals, start_block, owner, arg_count: body.params.len() };
    let mut ctx = MirLowerCtx {
        result: mir,
        db,
        infer,
        body,
        binding_locals,
        owner,
        current_loop_blocks: None,
        discr_temp: None,
    };
    let b = ctx.lower_expr_to_place(root_expr, return_slot().into(), start_block)?;
    ctx.result.basic_blocks[b].terminator = Some(Terminator::Return);
    Ok(ctx.result)
}
