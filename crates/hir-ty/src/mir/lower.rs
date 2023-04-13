//! This module generates a polymorphic MIR from a hir body

use std::{iter, mem, sync::Arc};

use chalk_ir::{BoundVar, ConstData, DebruijnIndex, TyKind};
use hir_def::{
    body::Body,
    data::adt::{StructKind, VariantData},
    hir::{
        Array, BindingAnnotation, BindingId, ExprId, LabelId, Literal, MatchArm, Pat, PatId,
        RecordFieldPat, RecordLitField,
    },
    lang_item::{LangItem, LangItemTarget},
    layout::LayoutError,
    path::Path,
    resolver::{resolver_for_expr, ResolveValueResult, ValueNs},
    AdtId, DefWithBodyId, EnumVariantId, HasModule, ItemContainerId, LocalFieldId, TraitId,
};
use hir_expand::name::Name;
use la_arena::ArenaMap;
use rustc_hash::FxHashMap;

use crate::{
    consteval::ConstEvalError,
    db::HirDatabase,
    display::HirDisplay,
    infer::{CaptureKind, CapturedItem, TypeMismatch},
    inhabitedness::is_ty_uninhabited_from,
    layout::layout_of_ty,
    mapping::ToChalk,
    static_lifetime,
    utils::generics,
    Adjust, Adjustment, AutoBorrow, CallableDefId, TyBuilder, TyExt,
};

use super::*;

mod as_place;
mod pattern_matching;

use pattern_matching::AdtPatternShape;

#[derive(Debug, Clone)]
struct LoopBlocks {
    begin: BasicBlockId,
    /// `None` for loops that are not terminating
    end: Option<BasicBlockId>,
    place: Place,
}

struct MirLowerCtx<'a> {
    result: MirBody,
    owner: DefWithBodyId,
    current_loop_blocks: Option<LoopBlocks>,
    // FIXME: we should resolve labels in HIR lowering and always work with label id here, not
    // with raw names.
    labeled_loop_blocks: FxHashMap<LabelId, LoopBlocks>,
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
    /// Trying to lower a trait function, instead of an implementation
    TraitFunctionDefinition(TraitId, Name),
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
    ImplementationError(String),
    LangItemNotFound(LangItem),
    MutatingRvalue,
    UnresolvedLabel,
    UnresolvedUpvar(Place),
    UnaccessableLocal,
}

macro_rules! not_supported {
    ($x: expr) => {
        return Err(MirLowerError::NotSupported(format!($x)))
    };
}

macro_rules! implementation_error {
    ($x: expr) => {{
        ::stdx::never!("MIR lower implementation bug: {}", format!($x));
        return Err(MirLowerError::ImplementationError(format!($x)));
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

impl<'ctx> MirLowerCtx<'ctx> {
    fn new(
        db: &'ctx dyn HirDatabase,
        owner: DefWithBodyId,
        body: &'ctx Body,
        infer: &'ctx InferenceResult,
    ) -> Self {
        let mut basic_blocks = Arena::new();
        let start_block = basic_blocks.alloc(BasicBlock {
            statements: vec![],
            terminator: None,
            is_cleanup: false,
        });
        let locals = Arena::new();
        let binding_locals: ArenaMap<BindingId, LocalId> = ArenaMap::new();
        let mir = MirBody {
            basic_blocks,
            locals,
            start_block,
            binding_locals,
            param_locals: vec![],
            owner,
            arg_count: body.params.len(),
            closures: vec![],
        };
        let ctx = MirLowerCtx {
            result: mir,
            db,
            infer,
            body,
            owner,
            current_loop_blocks: None,
            labeled_loop_blocks: Default::default(),
            discr_temp: None,
        };
        ctx
    }

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
            Expr::Missing => {
                if let DefWithBodyId::FunctionId(f) = self.owner {
                    let assoc = self.db.lookup_intern_function(f);
                    if let ItemContainerId::TraitId(t) = assoc.container {
                        let name = &self.db.function_data(f).name;
                        return Err(MirLowerError::TraitFunctionDefinition(t, name.clone()));
                    }
                }
                Err(MirLowerError::IncompleteExpr)
            },
            Expr::Path(p) => {
                let unresolved_name = || MirLowerError::unresolved_path(self.db, p);
                let resolver = resolver_for_expr(self.db.upcast(), self.owner, expr_id);
                let pr = resolver
                    .resolve_path_in_value_ns(self.db.upcast(), p)
                    .ok_or_else(unresolved_name)?;
                let pr = match pr {
                    ResolveValueResult::ValueNs(v) => v,
                    ResolveValueResult::Partial(..) => {
                        if let Some((assoc, subst)) = self
                            .infer
                            .assoc_resolutions_for_expr(expr_id)
                        {
                            match assoc {
                                hir_def::AssocItemId::ConstId(c) => {
                                    self.lower_const(c, current, place, subst, expr_id.into())?;
                                    return Ok(Some(current))
                                },
                                hir_def::AssocItemId::FunctionId(_) => {
                                    // FnDefs are zero sized, no action is needed.
                                    return Ok(Some(current))
                                }
                                hir_def::AssocItemId::TypeAliasId(_) => {
                                    // FIXME: If it is unreachable, use proper error instead of `not_supported`.
                                    not_supported!("associated functions and types")
                                },
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
                            Operand::Copy(self.binding_local(pat_id)?.into()).into(),
                            expr_id.into(),
                        );
                        Ok(Some(current))
                    }
                    ValueNs::ConstId(const_id) => {
                        self.lower_const(const_id, current, place, Substitution::empty(Interner), expr_id.into())?;
                        Ok(Some(current))
                    }
                    ValueNs::EnumVariantId(variant_id) => {
                        let variant_data = &self.db.enum_data(variant_id.parent).variants[variant_id.local_id];
                        if variant_data.variant_data.kind() == StructKind::Unit {
                            let ty = self.infer.type_of_expr[expr_id].clone();
                            current = self.lower_enum_variant(
                                variant_id,
                                current,
                                place,
                                ty,
                                vec![],
                                expr_id.into(),
                            )?;
                        }
                        // Otherwise its a tuple like enum, treated like a zero sized function, so no action is needed
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
                    ValueNs::FunctionId(_) | ValueNs::StructId(_) => {
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
                self.lower_block_to_place(statements, current, *tail, place)
            }
            Expr::Block { id: _, statements, tail, label } => {
                if let Some(label) = label {
                    self.lower_loop(current, place.clone(), Some(*label), |this, begin| {
                        if let Some(block) = this.lower_block_to_place(statements, begin, *tail, place)? {
                            let end = this.current_loop_end()?;
                            this.set_goto(block, end);
                        }
                        Ok(())
                    })
                } else {
                    self.lower_block_to_place(statements, current, *tail, place)
                }
            }
            Expr::Loop { body, label } => self.lower_loop(current, place, *label, |this, begin| {
                if let Some((_, block)) = this.lower_expr_as_place(begin, *body, true)? {
                    this.set_goto(block, begin);
                }
                Ok(())
            }),
            Expr::While { condition, body, label } => {
                self.lower_loop(current, place, *label, |this, begin| {
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
                self.lower_loop(current, place, label, |this, begin| {
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
                        AdtPatternShape::Tuple { args: &[pat], ellipsis: None },
                    )?;
                    if let Some((_, block)) = this.lower_expr_as_place(current, body, true)? {
                        this.set_goto(block, begin);
                    }
                    Ok(())
                })
            },
            Expr::Call { callee, args, .. } => {
                if let Some((func_id, generic_args)) =
                    self.infer.method_resolution(expr_id) {
                    let ty = chalk_ir::TyKind::FnDef(
                        CallableDefId::FunctionId(func_id).to_chalk(self.db),
                        generic_args,
                    )
                    .intern(Interner);
                    let func = Operand::from_bytes(vec![], ty);
                    return self.lower_call_and_args(
                        func,
                        iter::once(*callee).chain(args.iter().copied()),
                        place,
                        current,
                        self.is_uninhabited(expr_id),
                    );
                }
                let callee_ty = self.expr_ty_after_adjustments(*callee);
                match &callee_ty.data(Interner).kind {
                    chalk_ir::TyKind::FnDef(..) => {
                        let func = Operand::from_bytes(vec![], callee_ty.clone());
                        self.lower_call_and_args(func, args.iter().copied(), place, current, self.is_uninhabited(expr_id))
                    }
                    chalk_ir::TyKind::Function(_) => {
                        let Some((func, current)) = self.lower_expr_to_some_operand(*callee, current)? else {
                            return Ok(None);
                        };
                        self.lower_call_and_args(func, args.iter().copied(), place, current, self.is_uninhabited(expr_id))
                    }
                    TyKind::Error => return Err(MirLowerError::MissingFunctionDefinition),
                    _ => return Err(MirLowerError::TypeError("function call on bad type")),
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
                    let (then, mut otherwise) = self.pattern_match(
                        current,
                        None,
                        cond_place.clone(),
                        cond_ty.clone(),
                        *pat,
                        BindingAnnotation::Unannotated,
                    )?;
                    let then = if let &Some(guard) = guard {
                        let next = self.new_basic_block();
                        let o = otherwise.get_or_insert_with(|| self.new_basic_block());
                        if let Some((discr, c)) = self.lower_expr_to_some_operand(guard, then)? {
                            self.set_terminator(c, Terminator::SwitchInt { discr, targets: SwitchTargets::static_if(1, next, *o) });
                        }
                        next
                    } else {
                        then
                    };
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
                        self.current_loop_blocks.as_ref().ok_or(MirLowerError::ContinueWithoutLoop)?;
                    self.set_goto(current, loop_data.begin);
                    Ok(None)
                }
            },
            &Expr::Break { expr, label } => {
                if let Some(expr) = expr {
                    let loop_data = match label {
                        Some(l) => self.labeled_loop_blocks.get(&l).ok_or(MirLowerError::UnresolvedLabel)?,
                        None => self.current_loop_blocks.as_ref().ok_or(MirLowerError::BreakWithoutLoop)?,
                    };
                    let Some(c) = self.lower_expr_to_place(expr, loop_data.place.clone(), current)? else {
                        return Ok(None);
                    };
                    current = c;
                }
                let end = match label {
                    Some(l) => self.labeled_loop_blocks.get(&l).ok_or(MirLowerError::UnresolvedLabel)?.end.expect("We always generate end for labeled loops"),
                    None => self.current_loop_end()?,
                };
                self.set_goto(current, end);
                Ok(None)
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
            Expr::RecordLit { fields, path, spread, ellipsis: _, is_assignee_expr: _ } => {
                let spread_place = match spread {
                    &Some(x) => {
                        let Some((p, c)) = self.lower_expr_as_place(current, x, true)? else {
                            return Ok(None);
                        };
                        current = c;
                        Some(p)
                    },
                    None => None,
                };
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
                                match spread_place {
                                    Some(sp) => operands.into_iter().enumerate().map(|(i, x)| {
                                        match x {
                                            Some(x) => x,
                                            None => {
                                                let mut p = sp.clone();
                                                p.projection.push(ProjectionElem::Field(FieldId {
                                                    parent: variant_id,
                                                    local_id: LocalFieldId::from_raw(RawIdx::from(i as u32)),
                                                }));
                                                Operand::Copy(p)
                                            },
                                        }
                                    }).collect(),
                                    None => operands.into_iter().collect::<Option<_>>().ok_or(
                                        MirLowerError::TypeError("missing field in record literal"),
                                    )?,
                                },
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
            Expr::Yeet { .. } => not_supported!("yeet"),
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
            Expr::Field { .. } | Expr::Index { .. } | Expr::UnaryOp { op: hir_def::hir::UnaryOp::Deref, .. } => {
                let Some((p, current)) = self.lower_expr_as_place_without_adjust(current, expr_id, true)? else {
                    return Ok(None);
                };
                self.push_assignment(current, place, Operand::Copy(p).into(), expr_id.into());
                Ok(Some(current))
            }
            Expr::UnaryOp { expr, op: op @ (hir_def::hir::UnaryOp::Not | hir_def::hir::UnaryOp::Neg) } => {
                let Some((operand, current)) = self.lower_expr_to_some_operand(*expr, current)? else {
                    return Ok(None);
                };
                let operation = match op {
                    hir_def::hir::UnaryOp::Not => UnOp::Not,
                    hir_def::hir::UnaryOp::Neg => UnOp::Neg,
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
                if let hir_def::hir::BinaryOp::Assignment { op } = op {
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
                            hir_def::hir::BinaryOp::LogicOp(op) => match op {
                                hir_def::hir::LogicOp::And => BinOp::BitAnd, // FIXME: make these short circuit
                                hir_def::hir::LogicOp::Or => BinOp::BitOr,
                            },
                            hir_def::hir::BinaryOp::ArithOp(op) => BinOp::from(op),
                            hir_def::hir::BinaryOp::CmpOp(op) => BinOp::from(op),
                            hir_def::hir::BinaryOp::Assignment { .. } => unreachable!(), // handled above
                        },
                        lhs_op,
                        rhs_op,
                    ),
                    expr_id.into(),
                );
                Ok(Some(current))
            }
            &Expr::Range { lhs, rhs, range_type: _ } => {
                let ty = self.expr_ty(expr_id);
                let Some((adt, subst)) = ty.as_adt() else {
                    return Err(MirLowerError::TypeError("Range type is not adt"));
                };
                let AdtId::StructId(st) = adt else {
                    return Err(MirLowerError::TypeError("Range type is not struct"));
                };
                let mut lp = None;
                let mut rp = None;
                if let Some(x) = lhs {
                    let Some((o, c)) = self.lower_expr_to_some_operand(x, current)? else {
                        return Ok(None);
                    };
                    lp = Some(o);
                    current = c;
                }
                if let Some(x) = rhs {
                    let Some((o, c)) = self.lower_expr_to_some_operand(x, current)? else {
                        return Ok(None);
                    };
                    rp = Some(o);
                    current = c;
                }
                self.push_assignment(
                    current,
                    place,
                    Rvalue::Aggregate(
                        AggregateKind::Adt(st.into(), subst.clone()),
                        self.db.struct_data(st).variant_data.fields().iter().map(|x| {
                            let o = match x.1.name.as_str() {
                                Some("start") => lp.take(),
                                Some("end") => rp.take(),
                                Some("exhausted") => Some(Operand::from_bytes(vec![0], TyBuilder::bool())),
                                _ => None,
                            };
                            o.ok_or(MirLowerError::UnresolvedField)
                        }).collect::<Result<_>>()?,
                    ),
                    expr_id.into(),
                );
                Ok(Some(current))
            },
            Expr::Closure { .. } => {
                let ty = self.expr_ty(expr_id);
                let TyKind::Closure(id, _) = ty.kind(Interner) else {
                    not_supported!("closure with non closure type");
                };
                self.result.closures.push(*id);
                let (captures, _) = self.infer.closure_info(id);
                let mut operands = vec![];
                for capture in captures.iter() {
                    let p = Place {
                        local: self.binding_local(capture.place.local)?,
                        projection: capture.place.projections.clone().into_iter().map(|x| {
                            match x {
                                ProjectionElem::Deref => ProjectionElem::Deref,
                                ProjectionElem::Field(x) => ProjectionElem::Field(x),
                                ProjectionElem::TupleOrClosureField(x) => ProjectionElem::TupleOrClosureField(x),
                                ProjectionElem::ConstantIndex { offset, min_length, from_end } => ProjectionElem::ConstantIndex { offset, min_length, from_end },
                                ProjectionElem::Subslice { from, to, from_end } => ProjectionElem::Subslice { from, to, from_end },
                                ProjectionElem::OpaqueCast(x) => ProjectionElem::OpaqueCast(x),
                                ProjectionElem::Index(x) => match x { },
                            }
                        }).collect(),
                    };
                    match &capture.kind {
                        CaptureKind::ByRef(bk) => {
                            let tmp: Place = self.temp(capture.ty.clone())?.into();
                            self.push_assignment(
                                current,
                                tmp.clone(),
                                Rvalue::Ref(bk.clone(), p),
                                expr_id.into(),
                            );
                            operands.push(Operand::Move(tmp));
                        },
                        CaptureKind::ByValue => operands.push(Operand::Move(p)),
                    }
                }
                self.push_assignment(
                    current,
                    place,
                    Rvalue::Aggregate(AggregateKind::Closure(ty), operands),
                    expr_id.into(),
                );
                Ok(Some(current))
            },
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
                place.projection.push(ProjectionElem::TupleOrClosureField(index))
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
            hir_def::hir::Literal::String(b) => {
                let b = b.as_bytes();
                let mut data = vec![];
                data.extend(0usize.to_le_bytes());
                data.extend(b.len().to_le_bytes());
                let mut mm = MemoryMap::default();
                mm.insert(0, b.to_vec());
                return Ok(Operand::from_concrete_const(data, mm, ty));
            }
            hir_def::hir::Literal::ByteString(b) => {
                let mut data = vec![];
                data.extend(0usize.to_le_bytes());
                data.extend(b.len().to_le_bytes());
                let mut mm = MemoryMap::default();
                mm.insert(0, b.to_vec());
                return Ok(Operand::from_concrete_const(data, mm, ty));
            }
            hir_def::hir::Literal::Char(c) => u32::from(*c).to_le_bytes().into(),
            hir_def::hir::Literal::Bool(b) => vec![*b as u8],
            hir_def::hir::Literal::Int(x, _) => x.to_le_bytes()[0..size].into(),
            hir_def::hir::Literal::Uint(x, _) => x.to_le_bytes()[0..size].into(),
            hir_def::hir::Literal::Float(f, _) => match size {
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
        subst: Substitution,
        span: MirSpan,
    ) -> Result<()> {
        let c = self.db.const_eval(const_id, subst)?;
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
            _ => implementation_error!("Non ADT enum"),
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
        place: Place,
        label: Option<LabelId>,
        f: impl FnOnce(&mut MirLowerCtx<'_>, BasicBlockId) -> Result<()>,
    ) -> Result<Option<BasicBlockId>> {
        let begin = self.new_basic_block();
        let prev = mem::replace(
            &mut self.current_loop_blocks,
            Some(LoopBlocks { begin, end: None, place }),
        );
        let prev_label = if let Some(label) = label {
            // We should generate the end now, to make sure that it wouldn't change later. It is
            // bad as we may emit end (unnecessary unreachable block) for unterminating loop, but
            // it should not affect correctness.
            self.current_loop_end()?;
            self.labeled_loop_blocks
                .insert(label, self.current_loop_blocks.as_ref().unwrap().clone())
        } else {
            None
        };
        self.set_goto(prev_block, begin);
        f(self, begin)?;
        let my = mem::replace(&mut self.current_loop_blocks, prev).ok_or(
            MirLowerError::ImplementationError("current_loop_blocks is corrupt".to_string()),
        )?;
        if let Some(prev) = prev_label {
            self.labeled_loop_blocks.insert(label.unwrap(), prev);
        }
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
            .ok_or(MirLowerError::ImplementationError(
                "Current loop access out of loop".to_string(),
            ))?
            .end
        {
            Some(x) => x,
            None => {
                let s = self.new_basic_block();
                self.current_loop_blocks
                    .as_mut()
                    .ok_or(MirLowerError::ImplementationError(
                        "Current loop access out of loop".to_string(),
                    ))?
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
    fn push_storage_live(&mut self, b: BindingId, current: BasicBlockId) -> Result<()> {
        // Current implementation is wrong. It adds no `StorageDead` at the end of scope, and before each break
        // and continue. It just add a `StorageDead` before the `StorageLive`, which is not wrong, but unneeded in
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
        let l = self.binding_local(b)?;
        self.push_statement(current, StatementKind::StorageDead(l).with_span(span));
        self.push_statement(current, StatementKind::StorageLive(l).with_span(span));
        Ok(())
    }

    fn resolve_lang_item(&self, item: LangItem) -> Result<LangItemTarget> {
        let crate_id = self.owner.module(self.db.upcast()).krate();
        self.db.lang_item(crate_id, item).ok_or(MirLowerError::LangItemNotFound(item))
    }

    fn lower_block_to_place(
        &mut self,
        statements: &[hir_def::hir::Statement],
        mut current: BasicBlockId,
        tail: Option<ExprId>,
        place: Place,
    ) -> Result<Option<Idx<BasicBlock>>> {
        for statement in statements.iter() {
            match statement {
                hir_def::hir::Statement::Let { pat, initializer, else_branch, type_ref: _ } => {
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
                        let mut err = None;
                        self.body.walk_bindings_in_pat(*pat, |b| {
                            if let Err(e) = self.push_storage_live(b, current) {
                                err = Some(e);
                            }
                        });
                        if let Some(e) = err {
                            return Err(e);
                        }
                    }
                }
                hir_def::hir::Statement::Expr { expr, has_semi: _ } => {
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

    fn lower_params_and_bindings(
        &mut self,
        params: impl Iterator<Item = (PatId, Ty)> + Clone,
        pick_binding: impl Fn(BindingId) -> bool,
    ) -> Result<BasicBlockId> {
        let base_param_count = self.result.param_locals.len();
        self.result.param_locals.extend(params.clone().map(|(x, ty)| {
            let local_id = self.result.locals.alloc(Local { ty });
            if let Pat::Bind { id, subpat: None } = self.body[x] {
                if matches!(
                    self.body.bindings[id].mode,
                    BindingAnnotation::Unannotated | BindingAnnotation::Mutable
                ) {
                    self.result.binding_locals.insert(id, local_id);
                }
            }
            local_id
        }));
        // and then rest of bindings
        for (id, _) in self.body.bindings.iter() {
            if !pick_binding(id) {
                continue;
            }
            if !self.result.binding_locals.contains_idx(id) {
                self.result
                    .binding_locals
                    .insert(id, self.result.locals.alloc(Local { ty: self.infer[id].clone() }));
            }
        }
        let mut current = self.result.start_block;
        for ((param, _), local) in
            params.zip(self.result.param_locals.clone().into_iter().skip(base_param_count))
        {
            if let Pat::Bind { id, .. } = self.body[param] {
                if local == self.binding_local(id)? {
                    continue;
                }
            }
            let r = self.pattern_match(
                current,
                None,
                local.into(),
                self.result.locals[local].ty.clone(),
                param,
                BindingAnnotation::Unannotated,
            )?;
            if let Some(b) = r.1 {
                self.set_terminator(b, Terminator::Unreachable);
            }
            current = r.0;
        }
        Ok(current)
    }

    fn binding_local(&self, b: BindingId) -> Result<LocalId> {
        match self.result.binding_locals.get(b) {
            Some(x) => Ok(*x),
            None => Err(MirLowerError::UnaccessableLocal),
        }
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
        (TyKind::Scalar(_), TyKind::Raw(..)) => CastKind::PointerFromExposedAddress,
        (TyKind::Raw(..), TyKind::Scalar(_)) => CastKind::PointerExposeAddress,
        (TyKind::Raw(..) | TyKind::Ref(..), TyKind::Raw(..) | TyKind::Ref(..)) => {
            CastKind::PtrToPtr
        }
        // Enum to int casts
        (TyKind::Scalar(_), TyKind::Adt(..)) | (TyKind::Adt(..), TyKind::Scalar(_)) => {
            CastKind::IntToInt
        }
        (a, b) => not_supported!("Unknown cast between {a:?} and {b:?}"),
    })
}

pub fn mir_body_for_closure_query(
    db: &dyn HirDatabase,
    closure: ClosureId,
) -> Result<Arc<MirBody>> {
    let (owner, expr) = db.lookup_intern_closure(closure.into());
    let body = db.body(owner);
    let infer = db.infer(owner);
    let Expr::Closure { args, body: root, .. } = &body[expr] else {
        implementation_error!("closure expression is not closure");
    };
    let TyKind::Closure(_, substs) = &infer[expr].kind(Interner) else {
        implementation_error!("closure expression is not closure");
    };
    let (captures, _) = infer.closure_info(&closure);
    let mut ctx = MirLowerCtx::new(db, owner, &body, &infer);
    ctx.result.arg_count = args.len() + 1;
    // 0 is return local
    ctx.result.locals.alloc(Local { ty: infer[*root].clone() });
    ctx.result.locals.alloc(Local { ty: infer[expr].clone() });
    let Some(sig) = substs.at(Interner, 0).assert_ty_ref(Interner).callable_sig(db) else {
        implementation_error!("closure has not callable sig");
    };
    let current = ctx.lower_params_and_bindings(
        args.iter().zip(sig.params().iter()).map(|(x, y)| (*x, y.clone())),
        |_| true,
    )?;
    if let Some(b) = ctx.lower_expr_to_place(*root, return_slot().into(), current)? {
        ctx.set_terminator(b, Terminator::Return);
    }
    let mut upvar_map: FxHashMap<LocalId, Vec<(&CapturedItem, usize)>> = FxHashMap::default();
    for (i, capture) in captures.iter().enumerate() {
        let local = ctx.binding_local(capture.place.local)?;
        upvar_map.entry(local).or_default().push((capture, i));
    }
    let mut err = None;
    let closure_local = ctx.result.locals.iter().nth(1).unwrap().0;
    ctx.result.walk_places(|p| {
        if let Some(x) = upvar_map.get(&p.local) {
            let r = x.iter().find(|x| {
                if p.projection.len() < x.0.place.projections.len() {
                    return false;
                }
                for (x, y) in p.projection.iter().zip(x.0.place.projections.iter()) {
                    match (x, y) {
                        (ProjectionElem::Deref, ProjectionElem::Deref) => (),
                        (ProjectionElem::Field(x), ProjectionElem::Field(y)) if x == y => (),
                        (
                            ProjectionElem::TupleOrClosureField(x),
                            ProjectionElem::TupleOrClosureField(y),
                        ) if x == y => (),
                        _ => return false,
                    }
                }
                true
            });
            match r {
                Some(x) => {
                    p.local = closure_local;
                    let prev_projs =
                        mem::replace(&mut p.projection, vec![PlaceElem::TupleOrClosureField(x.1)]);
                    if x.0.kind != CaptureKind::ByValue {
                        p.projection.push(ProjectionElem::Deref);
                    }
                    p.projection.extend(prev_projs.into_iter().skip(x.0.place.projections.len()));
                }
                None => err = Some(p.clone()),
            }
        }
    });
    ctx.result.binding_locals = ctx
        .result
        .binding_locals
        .into_iter()
        .filter(|x| ctx.body[x.0].owner == Some(expr))
        .collect();
    if let Some(err) = err {
        return Err(MirLowerError::UnresolvedUpvar(err));
    }
    Ok(Arc::new(ctx.result))
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
    let mut ctx = MirLowerCtx::new(db, owner, body, infer);
    // 0 is return local
    ctx.result.locals.alloc(Local { ty: infer[root_expr].clone() });
    let binding_picker = |b: BindingId| {
        if root_expr == body.body_expr {
            body[b].owner.is_none()
        } else {
            body[b].owner == Some(root_expr)
        }
    };
    // 1 to param_len is for params
    let current = if let DefWithBodyId::FunctionId(fid) = owner {
        let substs = TyBuilder::placeholder_subst(db, fid);
        let callable_sig = db.callable_item_signature(fid.into()).substitute(Interner, &substs);
        ctx.lower_params_and_bindings(
            body.params.iter().zip(callable_sig.params().iter()).map(|(x, y)| (*x, y.clone())),
            binding_picker,
        )?
    } else {
        ctx.lower_params_and_bindings([].into_iter(), binding_picker)?
    };
    if let Some(b) = ctx.lower_expr_to_place(root_expr, return_slot().into(), current)? {
        ctx.set_terminator(b, Terminator::Return);
    }
    Ok(ctx.result)
}
