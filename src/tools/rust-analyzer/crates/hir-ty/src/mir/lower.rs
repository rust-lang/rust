//! This module generates a polymorphic MIR from a hir body

use std::{iter, mem, sync::Arc};

use chalk_ir::{BoundVar, ConstData, DebruijnIndex, TyKind};
use hir_def::{
    body::Body,
    expr::{
        Array, BindingAnnotation, BindingId, ExprId, LabelId, Literal, MatchArm, Pat, PatId,
        RecordLitField,
    },
    lang_item::{LangItem, LangItemTarget},
    layout::LayoutError,
    path::Path,
    resolver::{resolver_for_expr, ResolveValueResult, ValueNs},
    DefWithBodyId, EnumVariantId, HasModule,
};
use hir_expand::name::Name;
use la_arena::ArenaMap;

use crate::{
    consteval::ConstEvalError, db::HirDatabase, display::HirDisplay, infer::TypeMismatch,
    inhabitedness::is_ty_uninhabited_from, layout::layout_of_ty, mapping::ToChalk, static_lifetime,
    utils::generics, Adjust, Adjustment, AutoBorrow, CallableDefId, TyBuilder, TyExt,
};

use super::*;

mod as_place;

#[derive(Debug, Clone, Copy)]
struct LoopBlocks {
    begin: BasicBlockId,
    /// `None` for loops that are not terminating
    end: Option<BasicBlockId>,
}

struct MirLowerCtx<'a> {
    result: MirBody,
    owner: DefWithBodyId,
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
    UnresolvedName(String),
    RecordLiteralWithoutPath,
    UnresolvedMethod,
    UnresolvedField,
    MissingFunctionDefinition,
    TypeMismatch(TypeMismatch),
    /// This should be never happen. Type mismatch should catch everything.
    TypeError(&'static str),
    NotSupported(String),
    ContinueWithoutLoop,
    BreakWithoutLoop,
    Loop,
    /// Something that should never happen and is definitely a bug, but we don't want to panic if it happened
    ImplementationError(&'static str),
    LangItemNotFound(LangItem),
    MutatingRvalue,
}

macro_rules! not_supported {
    ($x: expr) => {
        return Err(MirLowerError::NotSupported(format!($x)))
    };
}

macro_rules! implementation_error {
    ($x: expr) => {{
        ::stdx::never!("MIR lower implementation bug: {}", $x);
        return Err(MirLowerError::ImplementationError($x));
    }};
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

impl MirLowerError {
    fn unresolved_path(db: &dyn HirDatabase, p: &Path) -> Self {
        Self::UnresolvedName(p.display(db).to_string())
    }
}

type Result<T> = std::result::Result<T, MirLowerError>;

impl MirLowerCtx<'_> {
    fn temp(&mut self, ty: Ty) -> Result<LocalId> {
        if matches!(ty.kind(Interner), TyKind::Slice(_) | TyKind::Dyn(_)) {
            implementation_error!("unsized temporaries");
        }
        Ok(self.result.locals.alloc(Local { ty }))
    }

    fn lower_expr_to_some_operand(
        &mut self,
        expr_id: ExprId,
        current: BasicBlockId,
    ) -> Result<Option<(Operand, BasicBlockId)>> {
        if !self.has_adjustments(expr_id) {
            match &self.body.exprs[expr_id] {
                Expr::Literal(l) => {
                    let ty = self.expr_ty(expr_id);
                    return Ok(Some((self.lower_literal_to_operand(ty, l)?, current)));
                }
                _ => (),
            }
        }
        let Some((p, current)) = self.lower_expr_as_place(current, expr_id, true)? else {
            return Ok(None);
        };
        Ok(Some((Operand::Copy(p), current)))
    }

    fn lower_expr_to_place_with_adjust(
        &mut self,
        expr_id: ExprId,
        place: Place,
        current: BasicBlockId,
        adjustments: &[Adjustment],
    ) -> Result<Option<BasicBlockId>> {
        match adjustments.split_last() {
            Some((last, rest)) => match &last.kind {
                Adjust::NeverToAny => {
                    let temp = self.temp(TyKind::Never.intern(Interner))?;
                    self.lower_expr_to_place_with_adjust(expr_id, temp.into(), current, rest)
                }
                Adjust::Deref(_) => {
                    let Some((p, current)) = self.lower_expr_as_place_with_adjust(current, expr_id, true, adjustments)? else {
                            return Ok(None);
                        };
                    self.push_assignment(current, place, Operand::Copy(p).into(), expr_id.into());
                    Ok(Some(current))
                }
                Adjust::Borrow(AutoBorrow::Ref(m) | AutoBorrow::RawPtr(m)) => {
                    let Some((p, current)) = self.lower_expr_as_place_with_adjust(current, expr_id, true, rest)? else {
                            return Ok(None);
                        };
                    let bk = BorrowKind::from_chalk(*m);
                    self.push_assignment(current, place, Rvalue::Ref(bk, p), expr_id.into());
                    Ok(Some(current))
                }
                Adjust::Pointer(cast) => {
                    let Some((p, current)) = self.lower_expr_as_place_with_adjust(current, expr_id, true, rest)? else {
                            return Ok(None);
                        };
                    self.push_assignment(
                        current,
                        place,
                        Rvalue::Cast(
                            CastKind::Pointer(cast.clone()),
                            Operand::Copy(p).into(),
                            last.target.clone(),
                        ),
                        expr_id.into(),
                    );
                    Ok(Some(current))
                }
            },
            None => self.lower_expr_to_place_without_adjust(expr_id, place, current),
        }
    }

    fn lower_expr_to_place(
        &mut self,
        expr_id: ExprId,
        place: Place,
        prev_block: BasicBlockId,
    ) -> Result<Option<BasicBlockId>> {
        if let Some(adjustments) = self.infer.expr_adjustments.get(&expr_id) {
            return self.lower_expr_to_place_with_adjust(expr_id, place, prev_block, adjustments);
        }
        self.lower_expr_to_place_without_adjust(expr_id, place, prev_block)
    }

    fn lower_expr_to_place_without_adjust(
        &mut self,
        expr_id: ExprId,
        place: Place,
        mut current: BasicBlockId,
    ) -> Result<Option<BasicBlockId>> {
        match &self.body.exprs[expr_id] {
            Expr::Missing => Err(MirLowerError::IncompleteExpr),
            Expr::Path(p) => {
                let unresolved_name = || MirLowerError::unresolved_path(self.db, p);
                let resolver = resolver_for_expr(self.db.upcast(), self.owner, expr_id);
                let pr = resolver
                    .resolve_path_in_value_ns(self.db.upcast(), p.mod_path())
                    .ok_or_else(unresolved_name)?;
                let pr = match pr {
                    ResolveValueResult::ValueNs(v) => v,
                    ResolveValueResult::Partial(..) => {
                        if let Some(assoc) = self
                            .infer
                            .assoc_resolutions_for_expr(expr_id)
                        {
                            match assoc.0 {
                                hir_def::AssocItemId::ConstId(c) => {
                                    self.lower_const(c, current, place, expr_id.into())?;
                                    return Ok(Some(current))
                                },
                                _ => not_supported!("associated functions and types"),
                            }
                        } else if let Some(variant) = self
                            .infer
                            .variant_resolution_for_expr(expr_id)
                        {
                            match variant {
                                VariantId::EnumVariantId(e) => ValueNs::EnumVariantId(e),
                                VariantId::StructId(s) => ValueNs::StructId(s),
                                VariantId::UnionId(_) => implementation_error!("Union variant as path"),
                            }
                        } else {
                            return Err(unresolved_name());
                        }
                    }
                };
                match pr {
                    ValueNs::LocalBinding(pat_id) => {
                        self.push_assignment(
                            current,
                            place,
                            Operand::Copy(self.result.binding_locals[pat_id].into()).into(),
                            expr_id.into(),
                        );
                        Ok(Some(current))
                    }
                    ValueNs::ConstId(const_id) => {
                        self.lower_const(const_id, current, place, expr_id.into())?;
                        Ok(Some(current))
                    }
                    ValueNs::EnumVariantId(variant_id) => {
                        let ty = self.infer.type_of_expr[expr_id].clone();
                        let current = self.lower_enum_variant(
                            variant_id,
                            current,
                            place,
                            ty,
                            vec![],
                            expr_id.into(),
                        )?;
                        Ok(Some(current))
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
                            expr_id.into(),
                        );
                        Ok(Some(current))
                    }
                    ValueNs::StructId(_) => {
                        // It's probably a unit struct or a zero sized function, so no action is needed.
                        Ok(Some(current))
                    }
                    x => {
                        not_supported!("unknown name {x:?} in value name space");
                    }
                }
            }
            Expr::If { condition, then_branch, else_branch } => {
                let Some((discr, current)) = self.lower_expr_to_some_operand(*condition, current)? else {
                    return Ok(None);
                };
                let start_of_then = self.new_basic_block();
                let end_of_then =
                    self.lower_expr_to_place(*then_branch, place.clone(), start_of_then)?;
                let start_of_else = self.new_basic_block();
                let end_of_else = if let Some(else_branch) = else_branch {
                    self.lower_expr_to_place(*else_branch, place, start_of_else)?
                } else {
                    Some(start_of_else)
                };
                self.set_terminator(
                    current,
                    Terminator::SwitchInt {
                        discr,
                        targets: SwitchTargets::static_if(1, start_of_then, start_of_else),
                    },
                );
                Ok(self.merge_blocks(end_of_then, end_of_else))
            }
            Expr::Let { pat, expr } => {
                let Some((cond_place, current)) = self.lower_expr_as_place(current, *expr, true)? else {
                    return Ok(None);
                };
                let (then_target, else_target) = self.pattern_match(
                    current,
                    None,
                    cond_place,
                    self.expr_ty_after_adjustments(*expr),
                    *pat,
                    BindingAnnotation::Unannotated,
                )?;
                self.write_bytes_to_place(
                    then_target,
                    place.clone(),
                    vec![1],
                    TyBuilder::bool(),
                    MirSpan::Unknown,
                )?;
                if let Some(else_target) = else_target {
                    self.write_bytes_to_place(
                        else_target,
                        place,
                        vec![0],
                        TyBuilder::bool(),
                        MirSpan::Unknown,
                    )?;
                }
                Ok(self.merge_blocks(Some(then_target), else_target))
            }
            Expr::Unsafe { id: _, statements, tail } => {
                self.lower_block_to_place(None, statements, current, *tail, place)
            }
            Expr::Block { id: _, statements, tail, label } => {
                self.lower_block_to_place(*label, statements, current, *tail, place)
            }
            Expr::Loop { body, label } => self.lower_loop(current, *label, |this, begin| {
                if let Some((_, block)) = this.lower_expr_as_place(begin, *body, true)? {
                    this.set_goto(block, begin);
                }
                Ok(())
            }),
            Expr::While { condition, body, label } => {
                self.lower_loop(current, *label, |this, begin| {
                    let Some((discr, to_switch)) = this.lower_expr_to_some_operand(*condition, begin)? else {
                        return Ok(());
                    };
                    let end = this.current_loop_end()?;
                    let after_cond = this.new_basic_block();
                    this.set_terminator(
                        to_switch,
                        Terminator::SwitchInt {
                            discr,
                            targets: SwitchTargets::static_if(1, after_cond, end),
                        },
                    );
                    if let Some((_, block)) = this.lower_expr_as_place(after_cond, *body, true)? {
                        this.set_goto(block, begin);
                    }
                    Ok(())
                })
            }
            &Expr::For { iterable, pat, body, label } => {
                let into_iter_fn = self.resolve_lang_item(LangItem::IntoIterIntoIter)?
                    .as_function().ok_or(MirLowerError::LangItemNotFound(LangItem::IntoIterIntoIter))?;
                let iter_next_fn = self.resolve_lang_item(LangItem::IteratorNext)?
                    .as_function().ok_or(MirLowerError::LangItemNotFound(LangItem::IteratorNext))?;
                let option_some = self.resolve_lang_item(LangItem::OptionSome)?
                    .as_enum_variant().ok_or(MirLowerError::LangItemNotFound(LangItem::OptionSome))?;
                let option = option_some.parent;
                let into_iter_fn_op = Operand::const_zst(
                    TyKind::FnDef(
                        self.db.intern_callable_def(CallableDefId::FunctionId(into_iter_fn)).into(),
                        Substitution::from1(Interner, self.expr_ty(iterable))
                    ).intern(Interner));
                let iter_next_fn_op = Operand::const_zst(
                    TyKind::FnDef(
                        self.db.intern_callable_def(CallableDefId::FunctionId(iter_next_fn)).into(),
                        Substitution::from1(Interner, self.expr_ty(iterable))
                    ).intern(Interner));
                let &Some(iterator_ty) = &self.infer.type_of_for_iterator.get(&expr_id) else {
                    return Err(MirLowerError::TypeError("unknown for loop iterator type"));
                };
                let ref_mut_iterator_ty = TyKind::Ref(Mutability::Mut, static_lifetime(), iterator_ty.clone()).intern(Interner);
                let item_ty = &self.infer.type_of_pat[pat];
                let option_item_ty = TyKind::Adt(chalk_ir::AdtId(option.into()), Substitution::from1(Interner, item_ty.clone())).intern(Interner);
                let iterator_place: Place = self.temp(iterator_ty.clone())?.into();
                let option_item_place: Place = self.temp(option_item_ty.clone())?.into();
                let ref_mut_iterator_place: Place = self.temp(ref_mut_iterator_ty)?.into();
                let Some(current) = self.lower_call_and_args(into_iter_fn_op, Some(iterable).into_iter(), iterator_place.clone(), current, false)?
                else {
                    return Ok(None);
                };
                self.push_assignment(current, ref_mut_iterator_place.clone(), Rvalue::Ref(BorrowKind::Mut { allow_two_phase_borrow: false }, iterator_place), expr_id.into());
                self.lower_loop(current, label, |this, begin| {
                    let Some(current) = this.lower_call(iter_next_fn_op, vec![Operand::Copy(ref_mut_iterator_place)], option_item_place.clone(), begin, false)?
                    else {
                        return Ok(());
                    };
                    let end = this.current_loop_end()?;
                    let (current, _) = this.pattern_matching_variant(
                        option_item_ty.clone(),
                        BindingAnnotation::Unannotated,
                        option_item_place.into(),
                        option_some.into(),
                        current,
                        pat.into(),
                        Some(end),
                        &[pat], &None)?;
                    if let Some((_, block)) = this.lower_expr_as_place(current, body, true)? {
                        this.set_goto(block, begin);
                    }
                    Ok(())
                })
            },
            Expr::Call { callee, args, .. } => {
                let callee_ty = self.expr_ty_after_adjustments(*callee);
                match &callee_ty.data(Interner).kind {
                    chalk_ir::TyKind::FnDef(..) => {
                        let func = Operand::from_bytes(vec![], callee_ty.clone());
                        self.lower_call_and_args(func, args.iter().copied(), place, current, self.is_uninhabited(expr_id))
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
                    self.infer.method_resolution(expr_id).ok_or(MirLowerError::UnresolvedMethod)?;
                let ty = chalk_ir::TyKind::FnDef(
                    CallableDefId::FunctionId(func_id).to_chalk(self.db),
                    generic_args,
                )
                .intern(Interner);
                let func = Operand::from_bytes(vec![], ty);
                self.lower_call_and_args(
                    func,
                    iter::once(*receiver).chain(args.iter().copied()),
                    place,
                    current,
                    self.is_uninhabited(expr_id),
                )
            }
            Expr::Match { expr, arms } => {
                let Some((cond_place, mut current)) = self.lower_expr_as_place(current, *expr, true)?
                else {
                    return Ok(None);
                };
                let cond_ty = self.expr_ty_after_adjustments(*expr);
                let mut end = None;
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
                    if let Some(block) = self.lower_expr_to_place(*expr, place.clone(), then)? {
                        let r = end.get_or_insert_with(|| self.new_basic_block());
                        self.set_goto(block, *r);
                    }
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
                    Ok(None)
                }
            },
            Expr::Break { expr, label } => {
                if expr.is_some() {
                    not_supported!("break with value");
                }
                match label {
                    Some(_) => not_supported!("break with label"),
                    None => {
                        let end =
                            self.current_loop_end()?;
                        self.set_goto(current, end);
                        Ok(None)
                    }
                }
            }
            Expr::Return { expr } => {
                if let Some(expr) = expr {
                    if let Some(c) = self.lower_expr_to_place(*expr, return_slot().into(), current)? {
                        current = c;
                    } else {
                        return Ok(None);
                    }
                }
                self.set_terminator(current, Terminator::Return);
                Ok(None)
            }
            Expr::Yield { .. } => not_supported!("yield"),
            Expr::RecordLit { fields, path, .. } => {
                let variant_id = self
                    .infer
                    .variant_resolution_for_expr(expr_id)
                    .ok_or_else(|| match path {
                        Some(p) => MirLowerError::UnresolvedName(p.display(self.db).to_string()),
                        None => MirLowerError::RecordLiteralWithoutPath,
                    })?;
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
                                variant_data.field(name).ok_or(MirLowerError::UnresolvedField)?;
                            let Some((op, c)) = self.lower_expr_to_some_operand(*expr, current)? else {
                                return Ok(None);
                            };
                            current = c;
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
                            expr_id.into(),
                        );
                        Ok(Some(current))
                    }
                    VariantId::UnionId(union_id) => {
                        let [RecordLitField { name, expr }] = fields.as_ref() else {
                            not_supported!("Union record literal with more than one field");
                        };
                        let local_id =
                            variant_data.field(name).ok_or(MirLowerError::UnresolvedField)?;
                        let mut place = place;
                        place
                            .projection
                            .push(PlaceElem::Field(FieldId { parent: union_id.into(), local_id }));
                        self.lower_expr_to_place(*expr, place, current)
                    }
                }
            }
            Expr::Await { .. } => not_supported!("await"),
            Expr::Try { .. } => not_supported!("? operator"),
            Expr::Yeet { .. } => not_supported!("yeet"),
            Expr::TryBlock { .. } => not_supported!("try block"),
            Expr::Async { .. } => not_supported!("async block"),
            Expr::Const { .. } => not_supported!("anonymous const block"),
            Expr::Cast { expr, type_ref: _ } => {
                let Some((x, current)) = self.lower_expr_to_some_operand(*expr, current)? else {
                    return Ok(None);
                };
                let source_ty = self.infer[*expr].clone();
                let target_ty = self.infer[expr_id].clone();
                self.push_assignment(
                    current,
                    place,
                    Rvalue::Cast(cast_kind(&source_ty, &target_ty)?, x, target_ty),
                    expr_id.into(),
                );
                Ok(Some(current))
            }
            Expr::Ref { expr, rawness: _, mutability } => {
                let Some((p, current)) = self.lower_expr_as_place(current, *expr, true)? else {
                    return Ok(None);
                };
                let bk = BorrowKind::from_hir(*mutability);
                self.push_assignment(current, place, Rvalue::Ref(bk, p), expr_id.into());
                Ok(Some(current))
            }
            Expr::Box { .. } => not_supported!("box expression"),
            Expr::Field { .. } | Expr::Index { .. } | Expr::UnaryOp { op: hir_def::expr::UnaryOp::Deref, .. } => {
                let Some((p, current)) = self.lower_expr_as_place_without_adjust(current, expr_id, true)? else {
                    return Ok(None);
                };
                self.push_assignment(current, place, Operand::Copy(p).into(), expr_id.into());
                Ok(Some(current))
            }
            Expr::UnaryOp { expr, op: op @ (hir_def::expr::UnaryOp::Not | hir_def::expr::UnaryOp::Neg) } => {
                let Some((operand, current)) = self.lower_expr_to_some_operand(*expr, current)? else {
                    return Ok(None);
                };
                let operation = match op {
                    hir_def::expr::UnaryOp::Not => UnOp::Not,
                    hir_def::expr::UnaryOp::Neg => UnOp::Neg,
                    _ => unreachable!(),
                };
                self.push_assignment(
                    current,
                    place,
                    Rvalue::UnaryOp(operation, operand),
                    expr_id.into(),
                );
                Ok(Some(current))
            },
            Expr::BinaryOp { lhs, rhs, op } => {
                let op = op.ok_or(MirLowerError::IncompleteExpr)?;
                if let hir_def::expr::BinaryOp::Assignment { op } = op {
                    if op.is_some() {
                        not_supported!("assignment with arith op (like +=)");
                    }
                    let Some((lhs_place, current)) =
                        self.lower_expr_as_place(current, *lhs, false)?
                    else {
                        return Ok(None);
                    };
                    let Some((rhs_op, current)) = self.lower_expr_to_some_operand(*rhs, current)? else {
                        return Ok(None);
                    };
                    self.push_assignment(current, lhs_place, rhs_op.into(), expr_id.into());
                    return Ok(Some(current));
                }
                let Some((lhs_op, current)) = self.lower_expr_to_some_operand(*lhs, current)? else {
                    return Ok(None);
                };
                let Some((rhs_op, current)) = self.lower_expr_to_some_operand(*rhs, current)? else {
                    return Ok(None);
                };
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
                    expr_id.into(),
                );
                Ok(Some(current))
            }
            Expr::Range { .. } => not_supported!("range"),
            Expr::Closure { .. } => not_supported!("closure"),
            Expr::Tuple { exprs, is_assignee_expr: _ } => {
                let Some(values) = exprs
                        .iter()
                        .map(|x| {
                            let Some((o, c)) = self.lower_expr_to_some_operand(*x, current)? else {
                                return Ok(None);
                            };
                            current = c;
                            Ok(Some(o))
                        })
                        .collect::<Result<Option<_>>>()?
                else {
                    return Ok(None);
                };
                let r = Rvalue::Aggregate(
                    AggregateKind::Tuple(self.expr_ty(expr_id)),
                    values,
                );
                self.push_assignment(current, place, r, expr_id.into());
                Ok(Some(current))
            }
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
                    let Some(values) = elements
                            .iter()
                            .map(|x| {
                                let Some((o, c)) = self.lower_expr_to_some_operand(*x, current)? else {
                                    return Ok(None);
                                };
                                current = c;
                                Ok(Some(o))
                            })
                            .collect::<Result<Option<_>>>()?
                    else {
                        return Ok(None);
                    };
                    let r = Rvalue::Aggregate(
                        AggregateKind::Array(elem_ty),
                        values,
                    );
                    self.push_assignment(current, place, r, expr_id.into());
                    Ok(Some(current))
                }
                Array::Repeat { .. } => not_supported!("array repeat"),
            },
            Expr::Literal(l) => {
                let ty = self.expr_ty(expr_id);
                let op = self.lower_literal_to_operand(ty, l)?;
                self.push_assignment(current, place, op.into(), expr_id.into());
                Ok(Some(current))
            }
            Expr::Underscore => not_supported!("underscore"),
        }
    }

    fn push_field_projection(&self, place: &mut Place, expr_id: ExprId) -> Result<()> {
        if let Expr::Field { expr, name } = &self.body[expr_id] {
            if let TyKind::Tuple(..) = self.expr_ty_after_adjustments(*expr).kind(Interner) {
                let index = name
                    .as_tuple_index()
                    .ok_or(MirLowerError::TypeError("named field on tuple"))?;
                place.projection.push(ProjectionElem::TupleField(index))
            } else {
                let field =
                    self.infer.field_resolution(expr_id).ok_or(MirLowerError::UnresolvedField)?;
                place.projection.push(ProjectionElem::Field(field));
            }
        } else {
            not_supported!("")
        }
        Ok(())
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
        span: MirSpan,
    ) -> Result<()> {
        let c = self.db.const_eval(const_id)?;
        self.write_const_to_place(c, prev_block, place, span)
    }

    fn write_const_to_place(
        &mut self,
        c: Const,
        prev_block: BasicBlockId,
        place: Place,
        span: MirSpan,
    ) -> Result<()> {
        self.push_assignment(prev_block, place, Operand::Constant(c).into(), span);
        Ok(())
    }

    fn write_bytes_to_place(
        &mut self,
        prev_block: BasicBlockId,
        place: Place,
        cv: Vec<u8>,
        ty: Ty,
        span: MirSpan,
    ) -> Result<()> {
        self.push_assignment(prev_block, place, Operand::from_bytes(cv, ty).into(), span);
        Ok(())
    }

    fn lower_enum_variant(
        &mut self,
        variant_id: EnumVariantId,
        prev_block: BasicBlockId,
        place: Place,
        ty: Ty,
        fields: Vec<Operand>,
        span: MirSpan,
    ) -> Result<BasicBlockId> {
        let subst = match ty.kind(Interner) {
            TyKind::Adt(_, subst) => subst.clone(),
            _ => not_supported!("Non ADT enum"),
        };
        self.push_assignment(
            prev_block,
            place,
            Rvalue::Aggregate(AggregateKind::Adt(variant_id.into(), subst), fields),
            span,
        );
        Ok(prev_block)
    }

    fn lower_call_and_args(
        &mut self,
        func: Operand,
        args: impl Iterator<Item = ExprId>,
        place: Place,
        mut current: BasicBlockId,
        is_uninhabited: bool,
    ) -> Result<Option<BasicBlockId>> {
        let Some(args) = args
            .map(|arg| {
                if let Some((temp, c)) = self.lower_expr_to_some_operand(arg, current)? {
                    current = c;
                    Ok(Some(temp))
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Option<Vec<_>>>>()?
        else {
            return Ok(None);
        };
        self.lower_call(func, args, place, current, is_uninhabited)
    }

    fn lower_call(
        &mut self,
        func: Operand,
        args: Vec<Operand>,
        place: Place,
        current: BasicBlockId,
        is_uninhabited: bool,
    ) -> Result<Option<BasicBlockId>> {
        let b = if is_uninhabited { None } else { Some(self.new_basic_block()) };
        self.set_terminator(
            current,
            Terminator::Call {
                func,
                args,
                destination: place,
                target: b,
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

    fn expr_ty_after_adjustments(&self, e: ExprId) -> Ty {
        let mut ty = None;
        if let Some(x) = self.infer.expr_adjustments.get(&e) {
            if let Some(x) = x.last() {
                ty = Some(x.target.clone());
            }
        }
        ty.unwrap_or_else(|| self.expr_ty(e))
    }

    fn push_statement(&mut self, block: BasicBlockId, statement: Statement) {
        self.result.basic_blocks[block].statements.push(statement);
    }

    fn push_assignment(
        &mut self,
        block: BasicBlockId,
        place: Place,
        rvalue: Rvalue,
        span: MirSpan,
    ) {
        self.push_statement(block, StatementKind::Assign(place, rvalue).with_span(span));
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
            Pat::Or(pats) => {
                let then_target = self.new_basic_block();
                let mut finished = false;
                for pat in &**pats {
                    let (next, next_else) = self.pattern_match(
                        current,
                        None,
                        cond_place.clone(),
                        cond_ty.clone(),
                        *pat,
                        binding_mode,
                    )?;
                    self.set_goto(next, then_target);
                    match next_else {
                        Some(t) => {
                            current = t;
                        }
                        None => {
                            finished = true;
                            break;
                        }
                    }
                }
                if !finished {
                    let ce = *current_else.get_or_insert_with(|| self.new_basic_block());
                    self.set_goto(current, ce);
                }
                (then_target, current_else)
            }
            Pat::Record { .. } => not_supported!("record pattern"),
            Pat::Range { .. } => not_supported!("range pattern"),
            Pat::Slice { .. } => not_supported!("slice pattern"),
            Pat::Path(_) => {
                let Some(variant) = self.infer.variant_resolution_for_pat(pattern) else {
                    not_supported!("unresolved variant");
                };
                self.pattern_matching_variant(
                    cond_ty,
                    binding_mode,
                    cond_place,
                    variant,
                    current,
                    pattern.into(),
                    current_else,
                    &[],
                    &None,
                )?
            }
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
            Pat::Bind { id, subpat } => {
                let target_place = self.result.binding_locals[*id];
                let mode = self.body.bindings[*id].mode;
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
                    binding_mode = mode;
                }
                self.push_storage_live(*id, current);
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
                    pattern.into(),
                );
                (current, current_else)
            }
            Pat::TupleStruct { path: _, args, ellipsis } => {
                let Some(variant) = self.infer.variant_resolution_for_pat(pattern) else {
                    not_supported!("unresolved variant");
                };
                self.pattern_matching_variant(
                    cond_ty,
                    binding_mode,
                    cond_place,
                    variant,
                    current,
                    pattern.into(),
                    current_else,
                    args,
                    ellipsis,
                )?
            }
            Pat::Ref { .. } => not_supported!("& pattern"),
            Pat::Box { .. } => not_supported!("box pattern"),
            Pat::ConstBlock(_) => not_supported!("const block pattern"),
        })
    }

    fn pattern_matching_variant(
        &mut self,
        mut cond_ty: Ty,
        mut binding_mode: BindingAnnotation,
        mut cond_place: Place,
        variant: VariantId,
        current: BasicBlockId,
        span: MirSpan,
        current_else: Option<BasicBlockId>,
        args: &[PatId],
        ellipsis: &Option<usize>,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        pattern_matching_dereference(&mut cond_ty, &mut binding_mode, &mut cond_place);
        let subst = match cond_ty.kind(Interner) {
            TyKind::Adt(_, s) => s,
            _ => return Err(MirLowerError::TypeError("non adt type matched with tuple struct")),
        };
        let fields_type = self.db.field_types(variant);
        Ok(match variant {
            VariantId::EnumVariantId(v) => {
                let e = self.db.const_eval_discriminant(v)? as u128;
                let next = self.new_basic_block();
                let tmp = self.discr_temp_place();
                self.push_assignment(
                    current,
                    tmp.clone(),
                    Rvalue::Discriminant(cond_place.clone()),
                    span,
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
                    enum_data.variants[v.local_id].variant_data.fields().iter().map(|(x, _)| {
                        (
                            PlaceElem::Field(FieldId { parent: v.into(), local_id: x }),
                            fields_type[x].clone().substitute(Interner, subst),
                        )
                    });
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
        f: impl FnOnce(&mut MirLowerCtx<'_>, BasicBlockId) -> Result<()>,
    ) -> Result<Option<BasicBlockId>> {
        if label.is_some() {
            not_supported!("loop with label");
        }
        let begin = self.new_basic_block();
        let prev =
            mem::replace(&mut self.current_loop_blocks, Some(LoopBlocks { begin, end: None }));
        self.set_goto(prev_block, begin);
        f(self, begin)?;
        let my = mem::replace(&mut self.current_loop_blocks, prev)
            .ok_or(MirLowerError::ImplementationError("current_loop_blocks is corrupt"))?;
        Ok(my.end)
    }

    fn has_adjustments(&self, expr_id: ExprId) -> bool {
        !self.infer.expr_adjustments.get(&expr_id).map(|x| x.is_empty()).unwrap_or(true)
    }

    fn merge_blocks(
        &mut self,
        b1: Option<BasicBlockId>,
        b2: Option<BasicBlockId>,
    ) -> Option<BasicBlockId> {
        match (b1, b2) {
            (None, None) => None,
            (None, Some(b)) | (Some(b), None) => Some(b),
            (Some(b1), Some(b2)) => {
                let bm = self.new_basic_block();
                self.set_goto(b1, bm);
                self.set_goto(b2, bm);
                Some(bm)
            }
        }
    }

    fn current_loop_end(&mut self) -> Result<BasicBlockId> {
        let r = match self
            .current_loop_blocks
            .as_mut()
            .ok_or(MirLowerError::ImplementationError("Current loop access out of loop"))?
            .end
        {
            Some(x) => x,
            None => {
                let s = self.new_basic_block();
                self.current_loop_blocks
                    .as_mut()
                    .ok_or(MirLowerError::ImplementationError("Current loop access out of loop"))?
                    .end = Some(s);
                s
            }
        };
        Ok(r)
    }

    fn is_uninhabited(&self, expr_id: ExprId) -> bool {
        is_ty_uninhabited_from(&self.infer[expr_id], self.owner.module(self.db.upcast()), self.db)
    }

    /// This function push `StorageLive` statement for the binding, and applies changes to add `StorageDead` in
    /// the appropriated places.
    fn push_storage_live(&mut self, b: BindingId, current: BasicBlockId) {
        // Current implementation is wrong. It adds no `StorageDead` at the end of scope, and before each break
        // and continue. It just add a `StorageDead` before the `StorageLive`, which is not wrong, but unneeeded in
        // the proper implementation. Due this limitation, implementing a borrow checker on top of this mir will falsely
        // allow this:
        //
        // ```
        // let x;
        // loop {
        //     let y = 2;
        //     x = &y;
        //     if some_condition {
        //         break; // we need to add a StorageDead(y) above this to kill the x borrow
        //     }
        // }
        // use(x)
        // ```
        // But I think this approach work for mutability analysis, as user can't write code which mutates a binding
        // after StorageDead, except loops, which are handled by this hack.
        let span = self.body.bindings[b]
            .definitions
            .first()
            .copied()
            .map(MirSpan::PatId)
            .unwrap_or(MirSpan::Unknown);
        let l = self.result.binding_locals[b];
        self.push_statement(current, StatementKind::StorageDead(l).with_span(span));
        self.push_statement(current, StatementKind::StorageLive(l).with_span(span));
    }

    fn resolve_lang_item(&self, item: LangItem) -> Result<LangItemTarget> {
        let crate_id = self.owner.module(self.db.upcast()).krate();
        self.db.lang_item(crate_id, item).ok_or(MirLowerError::LangItemNotFound(item))
    }

    fn lower_block_to_place(
        &mut self,
        label: Option<LabelId>,
        statements: &[hir_def::expr::Statement],
        mut current: BasicBlockId,
        tail: Option<ExprId>,
        place: Place,
    ) -> Result<Option<Idx<BasicBlock>>> {
        if label.is_some() {
            not_supported!("block with label");
        }
        for statement in statements.iter() {
            match statement {
                hir_def::expr::Statement::Let { pat, initializer, else_branch, type_ref: _ } => {
                    if let Some(expr_id) = initializer {
                        let else_block;
                        let Some((init_place, c)) =
                            self.lower_expr_as_place(current, *expr_id, true)?
                        else {
                            return Ok(None);
                        };
                        current = c;
                        (current, else_block) = self.pattern_match(
                            current,
                            None,
                            init_place,
                            self.expr_ty_after_adjustments(*expr_id),
                            *pat,
                            BindingAnnotation::Unannotated,
                        )?;
                        match (else_block, else_branch) {
                            (None, _) => (),
                            (Some(else_block), None) => {
                                self.set_terminator(else_block, Terminator::Unreachable);
                            }
                            (Some(else_block), Some(else_branch)) => {
                                if let Some((_, b)) =
                                    self.lower_expr_as_place(else_block, *else_branch, true)?
                                {
                                    self.set_terminator(b, Terminator::Unreachable);
                                }
                            }
                        }
                    } else {
                        self.body.walk_bindings_in_pat(*pat, |b| {
                            self.push_storage_live(b, current);
                        });
                    }
                }
                hir_def::expr::Statement::Expr { expr, has_semi: _ } => {
                    let Some((_, c)) = self.lower_expr_as_place(current, *expr, true)? else {
                        return Ok(None);
                    };
                    current = c;
                }
            }
        }
        match tail {
            Some(tail) => self.lower_expr_to_place(tail, place, current),
            None => Ok(Some(current)),
        }
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
    let _p = profile::span("mir_body_query").detail(|| match def {
        DefWithBodyId::FunctionId(it) => db.function_data(it).name.to_string(),
        DefWithBodyId::StaticId(it) => db.static_data(it).name.clone().to_string(),
        DefWithBodyId::ConstId(it) => {
            db.const_data(it).name.clone().unwrap_or_else(Name::missing).to_string()
        }
        DefWithBodyId::VariantId(it) => {
            db.enum_data(it.parent).variants[it.local_id].name.to_string()
        }
    });
    let body = db.body(def);
    let infer = db.infer(def);
    let result = lower_to_mir(db, def, &body, &infer, body.body_expr)?;
    Ok(Arc::new(result))
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
    if let Some((_, x)) = infer.type_mismatches().next() {
        return Err(MirLowerError::TypeMismatch(x.clone()));
    }
    let mut basic_blocks = Arena::new();
    let start_block =
        basic_blocks.alloc(BasicBlock { statements: vec![], terminator: None, is_cleanup: false });
    let mut locals = Arena::new();
    // 0 is return local
    locals.alloc(Local { ty: infer[root_expr].clone() });
    let mut binding_locals: ArenaMap<BindingId, LocalId> = ArenaMap::new();
    // 1 to param_len is for params
    let param_locals: Vec<LocalId> = if let DefWithBodyId::FunctionId(fid) = owner {
        let substs = TyBuilder::placeholder_subst(db, fid);
        let callable_sig = db.callable_item_signature(fid.into()).substitute(Interner, &substs);
        body.params
            .iter()
            .zip(callable_sig.params().iter())
            .map(|(&x, ty)| {
                let local_id = locals.alloc(Local { ty: ty.clone() });
                if let Pat::Bind { id, subpat: None } = body[x] {
                    if matches!(
                        body.bindings[id].mode,
                        BindingAnnotation::Unannotated | BindingAnnotation::Mutable
                    ) {
                        binding_locals.insert(id, local_id);
                    }
                }
                local_id
            })
            .collect()
    } else {
        if !body.params.is_empty() {
            return Err(MirLowerError::TypeError("Unexpected parameter for non function body"));
        }
        vec![]
    };
    // and then rest of bindings
    for (id, _) in body.bindings.iter() {
        if !binding_locals.contains_idx(id) {
            binding_locals.insert(id, locals.alloc(Local { ty: infer[id].clone() }));
        }
    }
    let mir = MirBody {
        basic_blocks,
        locals,
        start_block,
        binding_locals,
        param_locals,
        owner,
        arg_count: body.params.len(),
    };
    let mut ctx = MirLowerCtx {
        result: mir,
        db,
        infer,
        body,
        owner,
        current_loop_blocks: None,
        discr_temp: None,
    };
    let mut current = start_block;
    for (&param, local) in body.params.iter().zip(ctx.result.param_locals.clone().into_iter()) {
        if let Pat::Bind { id, .. } = body[param] {
            if local == ctx.result.binding_locals[id] {
                continue;
            }
        }
        let r = ctx.pattern_match(
            current,
            None,
            local.into(),
            ctx.result.locals[local].ty.clone(),
            param,
            BindingAnnotation::Unannotated,
        )?;
        if let Some(b) = r.1 {
            ctx.set_terminator(b, Terminator::Unreachable);
        }
        current = r.0;
    }
    if let Some(b) = ctx.lower_expr_to_place(root_expr, return_slot().into(), current)? {
        ctx.result.basic_blocks[b].terminator = Some(Terminator::Return);
    }
    Ok(ctx.result)
}
