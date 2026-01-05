//! This module generates a polymorphic MIR from a hir body

use std::{fmt::Write, iter, mem};

use base_db::Crate;
use hir_def::{
    AdtId, DefWithBodyId, EnumVariantId, GeneralConstId, GenericParamId, HasModule,
    ItemContainerId, LocalFieldId, Lookup, TraitId, TupleId,
    expr_store::{Body, ExpressionStore, HygieneId, path::Path},
    hir::{
        ArithOp, Array, BinaryOp, BindingAnnotation, BindingId, ExprId, LabelId, Literal, MatchArm,
        Pat, PatId, RecordFieldPat, RecordLitField, RecordSpread,
    },
    item_tree::FieldsShape,
    lang_item::LangItems,
    resolver::{HasResolver, ResolveValueResult, Resolver, ValueNs},
};
use hir_expand::name::Name;
use la_arena::ArenaMap;
use rustc_apfloat::Float;
use rustc_hash::FxHashMap;
use rustc_type_ir::inherent::{Const as _, GenericArgs as _, IntoKind, Ty as _};
use span::{Edition, FileId};
use syntax::TextRange;
use triomphe::Arc;

use crate::{
    Adjust, Adjustment, AutoBorrow, CallableDefId, ParamEnvAndCrate,
    consteval::ConstEvalError,
    db::{HirDatabase, InternedClosure, InternedClosureId},
    display::{DisplayTarget, HirDisplay, hir_display_with_store},
    generics::generics,
    infer::{
        CaptureKind, CapturedItem, TypeMismatch, cast::CastTy,
        closure::analysis::HirPlaceProjection,
    },
    inhabitedness::is_ty_uninhabited_from,
    layout::LayoutError,
    method_resolution::CandidateId,
    mir::{
        AggregateKind, Arena, BasicBlock, BasicBlockId, BinOp, BorrowKind, CastKind, Either, Expr,
        FieldId, GenericArgs, Idx, InferenceResult, Local, LocalId, MemoryMap, MirBody, MirSpan,
        Mutability, Operand, Place, PlaceElem, PointerCast, ProjectionElem, ProjectionStore,
        RawIdx, Rvalue, Statement, StatementKind, SwitchTargets, Terminator, TerminatorKind,
        TupleFieldId, Ty, UnOp, VariantId, return_slot,
    },
    next_solver::{
        Const, DbInterner, ParamConst, ParamEnv, Region, StoredGenericArgs, StoredTy, TyKind,
        TypingMode, UnevaluatedConst,
        abi::Safety,
        infer::{DbInternerInferExt, InferCtxt},
    },
    traits::FnTrait,
};

use super::OperandKind;

mod as_place;
mod pattern_matching;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
struct LoopBlocks {
    begin: BasicBlockId,
    /// `None` for loops that are not terminating
    end: Option<BasicBlockId>,
    place: Place,
    drop_scope_index: usize,
}

#[derive(Debug, Clone, Default)]
struct DropScope {
    /// locals, in order of definition (so we should run drop glues in reverse order)
    locals: Vec<LocalId>,
}

struct MirLowerCtx<'a, 'db> {
    result: MirBody,
    owner: DefWithBodyId,
    current_loop_blocks: Option<LoopBlocks>,
    labeled_loop_blocks: FxHashMap<LabelId, LoopBlocks>,
    discr_temp: Option<Place>,
    db: &'db dyn HirDatabase,
    body: &'a Body,
    infer: &'a InferenceResult,
    types: &'db crate::next_solver::DefaultAny<'db>,
    resolver: Resolver<'db>,
    drop_scopes: Vec<DropScope>,
    env: ParamEnv<'db>,
    infcx: InferCtxt<'db>,
}

// FIXME: Make this smaller, its stored in database queries
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MirLowerError {
    ConstEvalError(Box<str>, Box<ConstEvalError>),
    LayoutError(LayoutError),
    IncompleteExpr,
    IncompletePattern,
    /// Trying to lower a trait function, instead of an implementation
    TraitFunctionDefinition(TraitId, Name),
    UnresolvedName(String),
    RecordLiteralWithoutPath,
    UnresolvedMethod(String),
    UnresolvedField,
    UnsizedTemporary(StoredTy),
    MissingFunctionDefinition(DefWithBodyId, ExprId),
    TypeMismatch(TypeMismatch),
    HasErrors,
    /// This should never happen. Type mismatch should catch everything.
    TypeError(&'static str),
    NotSupported(String),
    ContinueWithoutLoop,
    BreakWithoutLoop,
    Loop,
    /// Something that should never happen and is definitely a bug, but we don't want to panic if it happened
    ImplementationError(String),
    LangItemNotFound,
    MutatingRvalue,
    UnresolvedLabel,
    UnresolvedUpvar(Place),
    InaccessibleLocal,

    // monomorphization errors:
    GenericArgNotProvided(GenericParamId, StoredGenericArgs),
}

/// A token to ensuring that each drop scope is popped at most once, thanks to the compiler that checks moves.
struct DropScopeToken;
impl DropScopeToken {
    fn pop_and_drop<'db>(
        self,
        ctx: &mut MirLowerCtx<'_, 'db>,
        current: BasicBlockId,
        span: MirSpan,
    ) -> BasicBlockId {
        std::mem::forget(self);
        ctx.pop_drop_scope_internal(current, span)
    }

    /// It is useful when we want a drop scope is syntactically closed, but we don't want to execute any drop
    /// code. Either when the control flow is diverging (so drop code doesn't reached) or when drop is handled
    /// for us (for example a block that ended with a return statement. Return will drop everything, so the block shouldn't
    /// do anything)
    fn pop_assume_dropped(self, ctx: &mut MirLowerCtx<'_, '_>) {
        std::mem::forget(self);
        ctx.pop_drop_scope_assume_dropped_internal();
    }
}

impl Drop for DropScopeToken {
    fn drop(&mut self) {}
}

// Uncomment this to make `DropScopeToken` a drop bomb. Unfortunately we can't do this in release, since
// in cases that mir lowering fails, we don't handle (and don't need to handle) drop scopes so it will be
// actually reached. `pop_drop_scope_assert_finished` will also detect this case, but doesn't show useful
// stack trace.
//
// impl Drop for DropScopeToken {
//     fn drop(&mut self) {
//         never!("Drop scope doesn't popped");
//     }
// }

impl MirLowerError {
    pub fn pretty_print(
        &self,
        f: &mut String,
        db: &dyn HirDatabase,
        span_formatter: impl Fn(FileId, TextRange) -> String,
        display_target: DisplayTarget,
    ) -> std::result::Result<(), std::fmt::Error> {
        match self {
            MirLowerError::ConstEvalError(name, e) => {
                writeln!(f, "In evaluating constant {name}")?;
                match &**e {
                    ConstEvalError::MirLowerError(e) => {
                        e.pretty_print(f, db, span_formatter, display_target)?
                    }
                    ConstEvalError::MirEvalError(e) => {
                        e.pretty_print(f, db, span_formatter, display_target)?
                    }
                }
            }
            MirLowerError::MissingFunctionDefinition(owner, it) => {
                let body = db.body(*owner);
                writeln!(
                    f,
                    "Missing function definition for {}",
                    body.pretty_print_expr(db, *owner, *it, display_target.edition)
                )?;
            }
            MirLowerError::HasErrors => writeln!(f, "Type inference result contains errors")?,
            MirLowerError::TypeMismatch(e) => writeln!(
                f,
                "Type mismatch: Expected {}, found {}",
                e.expected.as_ref().display(db, display_target),
                e.actual.as_ref().display(db, display_target),
            )?,
            MirLowerError::GenericArgNotProvided(id, subst) => {
                let param_name = match *id {
                    GenericParamId::TypeParamId(id) => {
                        db.generic_params(id.parent())[id.local_id()].name().cloned()
                    }
                    GenericParamId::ConstParamId(id) => {
                        db.generic_params(id.parent())[id.local_id()].name().cloned()
                    }
                    GenericParamId::LifetimeParamId(id) => {
                        Some(db.generic_params(id.parent)[id.local_id].name.clone())
                    }
                };
                writeln!(
                    f,
                    "Generic arg not provided for {}",
                    param_name.unwrap_or(Name::missing()).display(db, display_target.edition)
                )?;
                writeln!(f, "Provided args: [")?;
                for g in subst.as_ref() {
                    write!(f, "    {},", g.display(db, display_target))?;
                }
                writeln!(f, "]")?;
            }
            MirLowerError::LayoutError(_)
            | MirLowerError::UnsizedTemporary(_)
            | MirLowerError::IncompleteExpr
            | MirLowerError::IncompletePattern
            | MirLowerError::InaccessibleLocal
            | MirLowerError::TraitFunctionDefinition(_, _)
            | MirLowerError::UnresolvedName(_)
            | MirLowerError::RecordLiteralWithoutPath
            | MirLowerError::UnresolvedMethod(_)
            | MirLowerError::UnresolvedField
            | MirLowerError::TypeError(_)
            | MirLowerError::NotSupported(_)
            | MirLowerError::ContinueWithoutLoop
            | MirLowerError::BreakWithoutLoop
            | MirLowerError::Loop
            | MirLowerError::ImplementationError(_)
            | MirLowerError::LangItemNotFound
            | MirLowerError::MutatingRvalue
            | MirLowerError::UnresolvedLabel
            | MirLowerError::UnresolvedUpvar(_) => writeln!(f, "{self:?}")?,
        }
        Ok(())
    }
}

macro_rules! not_supported {
    ($it: expr) => {
        return Err(MirLowerError::NotSupported(format!($it)))
    };
}

macro_rules! implementation_error {
    ($it: expr) => {{
        ::stdx::never!("MIR lower implementation bug: {}", format!($it));
        return Err(MirLowerError::ImplementationError(format!($it)));
    }};
}

impl From<LayoutError> for MirLowerError {
    fn from(value: LayoutError) -> Self {
        MirLowerError::LayoutError(value)
    }
}

impl MirLowerError {
    fn unresolved_path(
        db: &dyn HirDatabase,
        p: &Path,
        display_target: DisplayTarget,
        store: &ExpressionStore,
    ) -> Self {
        Self::UnresolvedName(
            hir_display_with_store(p, store).display(db, display_target).to_string(),
        )
    }
}

type Result<'db, T> = std::result::Result<T, MirLowerError>;

impl<'a, 'db> MirLowerCtx<'a, 'db> {
    fn new(
        db: &'db dyn HirDatabase,
        owner: DefWithBodyId,
        body: &'a Body,
        infer: &'a InferenceResult,
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
            projection_store: ProjectionStore::default(),
            basic_blocks,
            locals,
            start_block,
            binding_locals,
            param_locals: vec![],
            owner,
            closures: vec![],
        };
        let resolver = owner.resolver(db);
        let env = db.trait_environment_for_body(owner);
        let interner = DbInterner::new_with(db, resolver.krate());
        // FIXME(next-solver): Is `non_body_analysis()` correct here? Don't we want to reveal opaque types defined by this body?
        let infcx = interner.infer_ctxt().build(TypingMode::non_body_analysis());

        MirLowerCtx {
            result: mir,
            db,
            infer,
            body,
            types: crate::next_solver::default_types(db),
            owner,
            resolver,
            current_loop_blocks: None,
            labeled_loop_blocks: Default::default(),
            discr_temp: None,
            drop_scopes: vec![DropScope::default()],
            env,
            infcx,
        }
    }

    #[inline]
    fn interner(&self) -> DbInterner<'db> {
        self.infcx.interner
    }

    #[inline]
    fn lang_items(&self) -> &'db LangItems {
        self.infcx.interner.lang_items()
    }

    fn temp(&mut self, ty: Ty<'db>, current: BasicBlockId, span: MirSpan) -> Result<'db, LocalId> {
        if matches!(ty.kind(), TyKind::Slice(_) | TyKind::Dynamic(..)) {
            return Err(MirLowerError::UnsizedTemporary(ty.store()));
        }
        let l = self.result.locals.alloc(Local { ty: ty.store() });
        self.push_storage_live_for_local(l, current, span)?;
        Ok(l)
    }

    fn lower_expr_to_some_operand(
        &mut self,
        expr_id: ExprId,
        current: BasicBlockId,
    ) -> Result<'db, Option<(Operand, BasicBlockId)>> {
        if !self.has_adjustments(expr_id)
            && let Expr::Literal(l) = &self.body[expr_id]
        {
            let ty = self.expr_ty_without_adjust(expr_id);
            return Ok(Some((self.lower_literal_to_operand(ty, l)?, current)));
        }
        let Some((p, current)) = self.lower_expr_as_place(current, expr_id, true)? else {
            return Ok(None);
        };
        Ok(Some((Operand { kind: OperandKind::Copy(p), span: Some(expr_id.into()) }, current)))
    }

    fn lower_expr_to_place_with_adjust(
        &mut self,
        expr_id: ExprId,
        place: Place,
        current: BasicBlockId,
        adjustments: &[Adjustment],
    ) -> Result<'db, Option<BasicBlockId>> {
        match adjustments.split_last() {
            Some((last, rest)) => match &last.kind {
                Adjust::NeverToAny => {
                    let temp = self.temp(self.types.types.never, current, MirSpan::Unknown)?;
                    self.lower_expr_to_place_with_adjust(expr_id, temp.into(), current, rest)
                }
                Adjust::Deref(_) => {
                    let Some((p, current)) =
                        self.lower_expr_as_place_with_adjust(current, expr_id, true, adjustments)?
                    else {
                        return Ok(None);
                    };
                    self.push_assignment(
                        current,
                        place,
                        Operand { kind: OperandKind::Copy(p), span: None }.into(),
                        expr_id.into(),
                    );
                    Ok(Some(current))
                }
                Adjust::Borrow(AutoBorrow::Ref(m)) => self.lower_expr_to_place_with_borrow_adjust(
                    expr_id,
                    place,
                    current,
                    rest,
                    (*m).into(),
                ),
                Adjust::Borrow(AutoBorrow::RawPtr(m)) => {
                    self.lower_expr_to_place_with_borrow_adjust(expr_id, place, current, rest, *m)
                }
                Adjust::Pointer(cast) => {
                    let Some((p, current)) =
                        self.lower_expr_as_place_with_adjust(current, expr_id, true, rest)?
                    else {
                        return Ok(None);
                    };
                    self.push_assignment(
                        current,
                        place,
                        Rvalue::Cast(
                            CastKind::PointerCoercion(*cast),
                            Operand { kind: OperandKind::Copy(p), span: None },
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

    fn lower_expr_to_place_with_borrow_adjust(
        &mut self,
        expr_id: ExprId,
        place: Place,
        current: BasicBlockId,
        rest: &[Adjustment],
        m: Mutability,
    ) -> Result<'db, Option<BasicBlockId>> {
        let Some((p, current)) =
            self.lower_expr_as_place_with_adjust(current, expr_id, true, rest)?
        else {
            return Ok(None);
        };
        let bk = BorrowKind::from_rustc(m);
        self.push_assignment(current, place, Rvalue::Ref(bk, p), expr_id.into());
        Ok(Some(current))
    }

    fn lower_expr_to_place(
        &mut self,
        expr_id: ExprId,
        place: Place,
        prev_block: BasicBlockId,
    ) -> Result<'db, Option<BasicBlockId>> {
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
    ) -> Result<'db, Option<BasicBlockId>> {
        match &self.body[expr_id] {
            Expr::OffsetOf(_) => {
                not_supported!("builtin#offset_of")
            }
            Expr::InlineAsm(_) => {
                not_supported!("builtin#asm")
            }
            Expr::Missing => {
                if let DefWithBodyId::FunctionId(f) = self.owner {
                    let assoc = f.lookup(self.db);
                    if let ItemContainerId::TraitId(t) = assoc.container {
                        let name = &self.db.function_signature(f).name;
                        return Err(MirLowerError::TraitFunctionDefinition(t, name.clone()));
                    }
                }
                Err(MirLowerError::IncompleteExpr)
            }
            Expr::Path(p) => {
                let pr =
                    if let Some((assoc, subst)) = self.infer.assoc_resolutions_for_expr(expr_id) {
                        match assoc {
                            CandidateId::ConstId(c) => {
                                self.lower_const(c.into(), current, place, subst, expr_id.into())?;
                                return Ok(Some(current));
                            }
                            CandidateId::FunctionId(_) => {
                                // FnDefs are zero sized, no action is needed.
                                return Ok(Some(current));
                            }
                        }
                    } else if let Some(variant) = self.infer.variant_resolution_for_expr(expr_id) {
                        match variant {
                            VariantId::EnumVariantId(e) => ValueNs::EnumVariantId(e),
                            VariantId::StructId(s) => ValueNs::StructId(s),
                            VariantId::UnionId(_) => implementation_error!("Union variant as path"),
                        }
                    } else {
                        let resolver_guard =
                            self.resolver.update_to_inner_scope(self.db, self.owner, expr_id);
                        let hygiene = self.body.expr_path_hygiene(expr_id);
                        let result = self
                            .resolver
                            .resolve_path_in_value_ns_fully(self.db, p, hygiene)
                            .ok_or_else(|| {
                                MirLowerError::unresolved_path(
                                    self.db,
                                    p,
                                    DisplayTarget::from_crate(self.db, self.krate()),
                                    self.body,
                                )
                            })?;
                        self.resolver.reset_to_guard(resolver_guard);
                        result
                    };
                match pr {
                    ValueNs::LocalBinding(_) | ValueNs::StaticId(_) => {
                        let Some((temp, current)) =
                            self.lower_expr_as_place_without_adjust(current, expr_id, false)?
                        else {
                            return Ok(None);
                        };
                        self.push_assignment(
                            current,
                            place,
                            Operand { kind: OperandKind::Copy(temp), span: None }.into(),
                            expr_id.into(),
                        );
                        Ok(Some(current))
                    }
                    ValueNs::ConstId(const_id) => {
                        self.lower_const(
                            const_id.into(),
                            current,
                            place,
                            GenericArgs::empty(self.interner()),
                            expr_id.into(),
                        )?;
                        Ok(Some(current))
                    }
                    ValueNs::EnumVariantId(variant_id) => {
                        let variant_fields = variant_id.fields(self.db);
                        if variant_fields.shape == FieldsShape::Unit {
                            let ty = self.infer.expr_ty(expr_id);
                            current = self.lower_enum_variant(
                                variant_id,
                                current,
                                place,
                                ty,
                                Box::new([]),
                                expr_id.into(),
                            )?;
                        }
                        // Otherwise its a tuple like enum, treated like a zero sized function, so no action is needed
                        Ok(Some(current))
                    }
                    ValueNs::GenericParam(p) => {
                        let Some(def) = self.owner.as_generic_def_id(self.db) else {
                            not_supported!("owner without generic def id");
                        };
                        let generics = generics(self.db, def);
                        let index = generics
                            .type_or_const_param_idx(p.into())
                            .ok_or(MirLowerError::TypeError("fail to lower const generic param"))?
                            as u32;
                        self.push_assignment(
                            current,
                            place,
                            Rvalue::from(Operand {
                                kind: OperandKind::Constant {
                                    konst: Const::new_param(
                                        self.interner(),
                                        ParamConst { id: p, index },
                                    )
                                    .store(),
                                    ty: self.db.const_param_ty_ns(p).store(),
                                },
                                span: None,
                            }),
                            expr_id.into(),
                        );
                        Ok(Some(current))
                    }
                    ValueNs::FunctionId(_) | ValueNs::StructId(_) | ValueNs::ImplSelf(_) => {
                        // It's probably a unit struct or a zero sized function, so no action is needed.
                        Ok(Some(current))
                    }
                }
            }
            Expr::If { condition, then_branch, else_branch } => {
                let Some((discr, current)) =
                    self.lower_expr_to_some_operand(*condition, current)?
                else {
                    return Ok(None);
                };
                let start_of_then = self.new_basic_block();
                let end_of_then = self.lower_expr_to_place(*then_branch, place, start_of_then)?;
                let start_of_else = self.new_basic_block();
                let end_of_else = if let Some(else_branch) = else_branch {
                    self.lower_expr_to_place(*else_branch, place, start_of_else)?
                } else {
                    Some(start_of_else)
                };
                self.set_terminator(
                    current,
                    TerminatorKind::SwitchInt {
                        discr,
                        targets: SwitchTargets::static_if(1, start_of_then, start_of_else),
                    },
                    expr_id.into(),
                );
                Ok(self.merge_blocks(end_of_then, end_of_else, expr_id.into()))
            }
            Expr::Let { pat, expr } => {
                let Some((cond_place, current)) = self.lower_expr_as_place(current, *expr, true)?
                else {
                    return Ok(None);
                };
                self.push_fake_read(current, cond_place, expr_id.into());
                let resolver_guard =
                    self.resolver.update_to_inner_scope(self.db, self.owner, expr_id);
                let (then_target, else_target) =
                    self.pattern_match(current, None, cond_place, *pat)?;
                self.resolver.reset_to_guard(resolver_guard);
                self.write_bytes_to_place(
                    then_target,
                    place,
                    Box::new([1]),
                    Ty::new_bool(self.interner()),
                    MirSpan::Unknown,
                )?;
                if let Some(else_target) = else_target {
                    self.write_bytes_to_place(
                        else_target,
                        place,
                        Box::new([0]),
                        Ty::new_bool(self.interner()),
                        MirSpan::Unknown,
                    )?;
                }
                Ok(self.merge_blocks(Some(then_target), else_target, expr_id.into()))
            }
            Expr::Unsafe { id: _, statements, tail } => {
                self.lower_block_to_place(statements, current, *tail, place, expr_id.into())
            }
            Expr::Block { id: _, statements, tail, label } => {
                if let Some(label) = label {
                    self.lower_loop(current, place, Some(*label), expr_id.into(), |this, begin| {
                        if let Some(current) = this.lower_block_to_place(
                            statements,
                            begin,
                            *tail,
                            place,
                            expr_id.into(),
                        )? {
                            let end = this.current_loop_end()?;
                            this.set_goto(current, end, expr_id.into());
                        }
                        Ok(())
                    })
                } else {
                    self.lower_block_to_place(statements, current, *tail, place, expr_id.into())
                }
            }
            Expr::Loop { body, label } => {
                self.lower_loop(current, place, *label, expr_id.into(), |this, begin| {
                    let scope = this.push_drop_scope();
                    if let Some((_, mut current)) = this.lower_expr_as_place(begin, *body, true)? {
                        current = scope.pop_and_drop(this, current, body.into());
                        this.set_goto(current, begin, expr_id.into());
                    } else {
                        scope.pop_assume_dropped(this);
                    }
                    Ok(())
                })
            }
            Expr::Call { callee, args, .. } => {
                if let Some((func_id, generic_args)) = self.infer.method_resolution(expr_id) {
                    let ty = Ty::new_fn_def(
                        self.interner(),
                        CallableDefId::FunctionId(func_id).into(),
                        generic_args,
                    );
                    let func = Operand::from_bytes(Box::default(), ty);
                    return self.lower_call_and_args(
                        func,
                        iter::once(*callee).chain(args.iter().copied()),
                        place,
                        current,
                        self.is_uninhabited(expr_id),
                        expr_id.into(),
                    );
                }
                let callee_ty = self.expr_ty_after_adjustments(*callee);
                match callee_ty.kind() {
                    TyKind::FnDef(..) => {
                        let func = Operand::from_bytes(Box::default(), callee_ty);
                        self.lower_call_and_args(
                            func,
                            args.iter().copied(),
                            place,
                            current,
                            self.is_uninhabited(expr_id),
                            expr_id.into(),
                        )
                    }
                    TyKind::FnPtr(..) => {
                        let Some((func, current)) =
                            self.lower_expr_to_some_operand(*callee, current)?
                        else {
                            return Ok(None);
                        };
                        self.lower_call_and_args(
                            func,
                            args.iter().copied(),
                            place,
                            current,
                            self.is_uninhabited(expr_id),
                            expr_id.into(),
                        )
                    }
                    TyKind::Closure(_, _) => {
                        not_supported!(
                            "method resolution not emitted for closure (Are Fn traits available?)"
                        );
                    }
                    TyKind::Error(_) => {
                        Err(MirLowerError::MissingFunctionDefinition(self.owner, expr_id))
                    }
                    _ => Err(MirLowerError::TypeError("function call on bad type")),
                }
            }
            Expr::MethodCall { receiver, args, method_name, .. } => {
                let (func_id, generic_args) =
                    self.infer.method_resolution(expr_id).ok_or_else(|| {
                        MirLowerError::UnresolvedMethod(
                            method_name.display(self.db, self.edition()).to_string(),
                        )
                    })?;
                let func = Operand::from_fn(self.db, func_id, generic_args);
                self.lower_call_and_args(
                    func,
                    iter::once(*receiver).chain(args.iter().copied()),
                    place,
                    current,
                    self.is_uninhabited(expr_id),
                    expr_id.into(),
                )
            }
            Expr::Match { expr, arms } => {
                let Some((cond_place, mut current)) =
                    self.lower_expr_as_place(current, *expr, true)?
                else {
                    return Ok(None);
                };
                self.push_fake_read(current, cond_place, expr_id.into());
                let mut end = None;
                let resolver_guard =
                    self.resolver.update_to_inner_scope(self.db, self.owner, expr_id);
                for MatchArm { pat, guard, expr } in arms.iter() {
                    let (then, mut otherwise) =
                        self.pattern_match(current, None, cond_place, *pat)?;
                    let then = if let &Some(guard) = guard {
                        let next = self.new_basic_block();
                        let o = otherwise.get_or_insert_with(|| self.new_basic_block());
                        if let Some((discr, c)) = self.lower_expr_to_some_operand(guard, then)? {
                            self.set_terminator(
                                c,
                                TerminatorKind::SwitchInt {
                                    discr,
                                    targets: SwitchTargets::static_if(1, next, *o),
                                },
                                expr_id.into(),
                            );
                        }
                        next
                    } else {
                        then
                    };
                    if let Some(block) = self.lower_expr_to_place(*expr, place, then)? {
                        let r = end.get_or_insert_with(|| self.new_basic_block());
                        self.set_goto(block, *r, expr_id.into());
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
                self.resolver.reset_to_guard(resolver_guard);
                if self.is_unterminated(current) {
                    self.set_terminator(current, TerminatorKind::Unreachable, expr_id.into());
                }
                Ok(end)
            }
            Expr::Continue { label } => {
                let loop_data = match label {
                    Some(l) => {
                        self.labeled_loop_blocks.get(l).ok_or(MirLowerError::UnresolvedLabel)?
                    }
                    None => self
                        .current_loop_blocks
                        .as_ref()
                        .ok_or(MirLowerError::ContinueWithoutLoop)?,
                };
                let begin = loop_data.begin;
                current =
                    self.drop_until_scope(loop_data.drop_scope_index, current, expr_id.into());
                self.set_goto(current, begin, expr_id.into());
                Ok(None)
            }
            &Expr::Break { expr, label } => {
                if let Some(expr) = expr {
                    let loop_data = match label {
                        Some(l) => self
                            .labeled_loop_blocks
                            .get(&l)
                            .ok_or(MirLowerError::UnresolvedLabel)?,
                        None => self
                            .current_loop_blocks
                            .as_ref()
                            .ok_or(MirLowerError::BreakWithoutLoop)?,
                    };
                    let Some(c) = self.lower_expr_to_place(expr, loop_data.place, current)? else {
                        return Ok(None);
                    };
                    current = c;
                }
                let (end, drop_scope) = match label {
                    Some(l) => {
                        let loop_blocks = self
                            .labeled_loop_blocks
                            .get(&l)
                            .ok_or(MirLowerError::UnresolvedLabel)?;
                        (
                            loop_blocks.end.expect("We always generate end for labeled loops"),
                            loop_blocks.drop_scope_index,
                        )
                    }
                    None => (
                        self.current_loop_end()?,
                        self.current_loop_blocks.as_ref().unwrap().drop_scope_index,
                    ),
                };
                current = self.drop_until_scope(drop_scope, current, expr_id.into());
                self.set_goto(current, end, expr_id.into());
                Ok(None)
            }
            Expr::Return { expr } => {
                if let Some(expr) = expr {
                    if let Some(c) =
                        self.lower_expr_to_place(*expr, return_slot().into(), current)?
                    {
                        current = c;
                    } else {
                        return Ok(None);
                    }
                }
                current = self.drop_until_scope(0, current, expr_id.into());
                self.set_terminator(current, TerminatorKind::Return, expr_id.into());
                Ok(None)
            }
            Expr::Become { .. } => not_supported!("tail-calls"),
            Expr::Yield { .. } => not_supported!("yield"),
            Expr::RecordLit { fields, path, spread, .. } => {
                let spread_place = match *spread {
                    RecordSpread::Expr(it) => {
                        let Some((p, c)) = self.lower_expr_as_place(current, it, true)? else {
                            return Ok(None);
                        };
                        current = c;
                        Some(p)
                    }
                    RecordSpread::None => None,
                    RecordSpread::FieldDefaults => not_supported!("empty record spread"),
                };
                let variant_id =
                    self.infer.variant_resolution_for_expr(expr_id).ok_or_else(|| match path {
                        Some(p) => MirLowerError::UnresolvedName(
                            hir_display_with_store(&**p, self.body)
                                .display(self.db, self.display_target())
                                .to_string(),
                        ),
                        None => MirLowerError::RecordLiteralWithoutPath,
                    })?;
                let subst = match self.expr_ty_without_adjust(expr_id).kind() {
                    TyKind::Adt(_, s) => s,
                    _ => not_supported!("Non ADT record literal"),
                };
                let variant_fields = variant_id.fields(self.db);
                match variant_id {
                    VariantId::EnumVariantId(_) | VariantId::StructId(_) => {
                        let mut operands = vec![None; variant_fields.fields().len()];
                        for RecordLitField { name, expr } in fields.iter() {
                            let field_id =
                                variant_fields.field(name).ok_or(MirLowerError::UnresolvedField)?;
                            let Some((op, c)) = self.lower_expr_to_some_operand(*expr, current)?
                            else {
                                return Ok(None);
                            };
                            current = c;
                            operands[u32::from(field_id.into_raw()) as usize] = Some(op);
                        }
                        let rvalue = Rvalue::Aggregate(
                            AggregateKind::Adt(variant_id, subst.store()),
                            match spread_place {
                                Some(sp) => operands
                                    .into_iter()
                                    .enumerate()
                                    .map(|(i, it)| match it {
                                        Some(it) => it,
                                        None => {
                                            let p = sp.project(
                                                ProjectionElem::Field(Either::Left(FieldId {
                                                    parent: variant_id,
                                                    local_id: LocalFieldId::from_raw(RawIdx::from(
                                                        i as u32,
                                                    )),
                                                })),
                                                &mut self.result.projection_store,
                                            );
                                            Operand { kind: OperandKind::Copy(p), span: None }
                                        }
                                    })
                                    .collect(),
                                None => operands.into_iter().collect::<Option<_>>().ok_or(
                                    MirLowerError::TypeError("missing field in record literal"),
                                )?,
                            },
                        );
                        self.push_assignment(current, place, rvalue, expr_id.into());
                        Ok(Some(current))
                    }
                    VariantId::UnionId(union_id) => {
                        let [RecordLitField { name, expr }] = fields.as_ref() else {
                            not_supported!("Union record literal with more than one field");
                        };
                        let local_id =
                            variant_fields.field(name).ok_or(MirLowerError::UnresolvedField)?;
                        let place = place.project(
                            PlaceElem::Field(Either::Left(FieldId {
                                parent: union_id.into(),
                                local_id,
                            })),
                            &mut self.result.projection_store,
                        );
                        self.lower_expr_to_place(*expr, place, current)
                    }
                }
            }
            Expr::Await { .. } => not_supported!("await"),
            Expr::Yeet { .. } => not_supported!("yeet"),
            Expr::Async { .. } => not_supported!("async block"),
            &Expr::Const(_) => {
                // let subst = self.placeholder_subst();
                // self.lower_const(
                //     id.into(),
                //     current,
                //     place,
                //     subst,
                //     expr_id.into(),
                //     self.expr_ty_without_adjust(expr_id),
                // )?;
                // Ok(Some(current))
                not_supported!("const block")
            }
            Expr::Cast { expr, type_ref: _ } => {
                let Some((it, current)) = self.lower_expr_to_some_operand(*expr, current)? else {
                    return Ok(None);
                };
                // Since we don't have THIR, this is the "zipped" version of [rustc's HIR lowering](https://github.com/rust-lang/rust/blob/e71f9529121ca8f687e4b725e3c9adc3f1ebab4d/compiler/rustc_mir_build/src/thir/cx/expr.rs#L165-L178)
                // and [THIR lowering as RValue](https://github.com/rust-lang/rust/blob/a4601859ae3875732797873612d424976d9e3dd0/compiler/rustc_mir_build/src/build/expr/as_rvalue.rs#L193-L313)
                let rvalue = if self.infer.coercion_casts.contains(expr) {
                    Rvalue::Use(it)
                } else {
                    let source_ty = self.infer.expr_ty(*expr);
                    let target_ty = self.infer.expr_ty(expr_id);
                    let cast_kind = if source_ty.as_reference().is_some() {
                        CastKind::PointerCoercion(PointerCast::ArrayToPointer)
                    } else {
                        cast_kind(self.db, source_ty, target_ty)?
                    };

                    Rvalue::Cast(cast_kind, it, target_ty.store())
                };
                self.push_assignment(current, place, rvalue, expr_id.into());
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
            Expr::Box { expr } => {
                let ty = self.expr_ty_after_adjustments(*expr);
                self.push_assignment(
                    current,
                    place,
                    Rvalue::ShallowInitBoxWithAlloc(ty.store()),
                    expr_id.into(),
                );
                let Some((operand, current)) = self.lower_expr_to_some_operand(*expr, current)?
                else {
                    return Ok(None);
                };
                let p = place.project(ProjectionElem::Deref, &mut self.result.projection_store);
                self.push_assignment(current, p, operand.into(), expr_id.into());
                Ok(Some(current))
            }
            Expr::Field { .. }
            | Expr::Index { .. }
            | Expr::UnaryOp { op: hir_def::hir::UnaryOp::Deref, .. } => {
                let Some((p, current)) =
                    self.lower_expr_as_place_without_adjust(current, expr_id, true)?
                else {
                    return Ok(None);
                };
                self.push_assignment(
                    current,
                    place,
                    Operand { kind: OperandKind::Copy(p), span: None }.into(),
                    expr_id.into(),
                );
                Ok(Some(current))
            }
            Expr::UnaryOp {
                expr,
                op: op @ (hir_def::hir::UnaryOp::Not | hir_def::hir::UnaryOp::Neg),
            } => {
                let Some((operand, current)) = self.lower_expr_to_some_operand(*expr, current)?
                else {
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
            }
            Expr::BinaryOp { lhs, rhs, op } => {
                let op: BinaryOp = op.ok_or(MirLowerError::IncompleteExpr)?;
                let is_builtin = 'b: {
                    // Without adjust here is a hack. We assume that we know every possible adjustment
                    // for binary operator, and use without adjust to simplify our conditions.
                    let lhs_ty = self.expr_ty_without_adjust(*lhs);
                    let rhs_ty = self.expr_ty_without_adjust(*rhs);
                    if matches!(op, BinaryOp::CmpOp(syntax::ast::CmpOp::Eq { .. }))
                        && matches!(lhs_ty.kind(), TyKind::RawPtr(..))
                        && matches!(rhs_ty.kind(), TyKind::RawPtr(..))
                    {
                        break 'b true;
                    }
                    let builtin_inequal_impls = matches!(
                        op,
                        BinaryOp::ArithOp(ArithOp::Shl | ArithOp::Shr)
                            | BinaryOp::Assignment { op: Some(ArithOp::Shl | ArithOp::Shr) }
                    );
                    matches!(
                        lhs_ty.kind(),
                        TyKind::Bool
                            | TyKind::Char
                            | TyKind::Int(_)
                            | TyKind::Uint(_)
                            | TyKind::Float(_)
                    ) && matches!(
                        rhs_ty.kind(),
                        TyKind::Bool
                            | TyKind::Char
                            | TyKind::Int(_)
                            | TyKind::Uint(_)
                            | TyKind::Float(_)
                    ) && (lhs_ty == rhs_ty || builtin_inequal_impls)
                };
                if !is_builtin
                    && let Some((func_id, generic_args)) = self.infer.method_resolution(expr_id)
                {
                    let func = Operand::from_fn(self.db, func_id, generic_args);
                    return self.lower_call_and_args(
                        func,
                        [*lhs, *rhs].into_iter(),
                        place,
                        current,
                        self.is_uninhabited(expr_id),
                        expr_id.into(),
                    );
                }
                if let hir_def::hir::BinaryOp::Assignment { op: Some(op) } = op {
                    // last adjustment is `&mut` which we don't want it.
                    let adjusts = self
                        .infer
                        .expr_adjustments
                        .get(lhs)
                        .and_then(|it| it.split_last())
                        .map(|it| it.1)
                        .ok_or(MirLowerError::TypeError("adjustment of binary op was missing"))?;
                    let Some((lhs_place, current)) =
                        self.lower_expr_as_place_with_adjust(current, *lhs, false, adjusts)?
                    else {
                        return Ok(None);
                    };
                    let Some((rhs_op, current)) = self.lower_expr_to_some_operand(*rhs, current)?
                    else {
                        return Ok(None);
                    };
                    let r_value = Rvalue::CheckedBinaryOp(
                        op.into(),
                        Operand { kind: OperandKind::Copy(lhs_place), span: None },
                        rhs_op,
                    );
                    self.push_assignment(current, lhs_place, r_value, expr_id.into());
                    return Ok(Some(current));
                }
                let Some((lhs_op, current)) = self.lower_expr_to_some_operand(*lhs, current)?
                else {
                    return Ok(None);
                };
                if let hir_def::hir::BinaryOp::LogicOp(op) = op {
                    let value_to_short = match op {
                        syntax::ast::LogicOp::And => 0,
                        syntax::ast::LogicOp::Or => 1,
                    };
                    let start_of_then = self.new_basic_block();
                    self.push_assignment(
                        start_of_then,
                        place,
                        lhs_op.clone().into(),
                        expr_id.into(),
                    );
                    let end_of_then = Some(start_of_then);
                    let start_of_else = self.new_basic_block();
                    let end_of_else = self.lower_expr_to_place(*rhs, place, start_of_else)?;
                    self.set_terminator(
                        current,
                        TerminatorKind::SwitchInt {
                            discr: lhs_op,
                            targets: SwitchTargets::static_if(
                                value_to_short,
                                start_of_then,
                                start_of_else,
                            ),
                        },
                        expr_id.into(),
                    );
                    return Ok(self.merge_blocks(end_of_then, end_of_else, expr_id.into()));
                }
                let Some((rhs_op, current)) = self.lower_expr_to_some_operand(*rhs, current)?
                else {
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
            &Expr::Assignment { target, value } => {
                let Some((value, mut current)) = self.lower_expr_as_place(current, value, true)?
                else {
                    return Ok(None);
                };
                self.push_fake_read(current, value, expr_id.into());
                let resolver_guard =
                    self.resolver.update_to_inner_scope(self.db, self.owner, expr_id);
                current = self.pattern_match_assignment(current, value, target)?;
                self.resolver.reset_to_guard(resolver_guard);
                Ok(Some(current))
            }
            &Expr::Range { lhs, rhs, range_type: _ } => {
                let ty = self.expr_ty_without_adjust(expr_id);
                let Some((adt, subst)) = ty.as_adt() else {
                    return Err(MirLowerError::TypeError("Range type is not adt"));
                };
                let AdtId::StructId(st) = adt else {
                    return Err(MirLowerError::TypeError("Range type is not struct"));
                };
                let mut lp = None;
                let mut rp = None;
                if let Some(it) = lhs {
                    let Some((o, c)) = self.lower_expr_to_some_operand(it, current)? else {
                        return Ok(None);
                    };
                    lp = Some(o);
                    current = c;
                }
                if let Some(it) = rhs {
                    let Some((o, c)) = self.lower_expr_to_some_operand(it, current)? else {
                        return Ok(None);
                    };
                    rp = Some(o);
                    current = c;
                }
                self.push_assignment(
                    current,
                    place,
                    Rvalue::Aggregate(
                        AggregateKind::Adt(st.into(), subst.store()),
                        st.fields(self.db)
                            .fields()
                            .iter()
                            .map(|it| {
                                let o = match it.1.name.as_str() {
                                    "start" => lp.take(),
                                    "end" => rp.take(),
                                    "exhausted" => Some(Operand::from_bytes(
                                        Box::new([0]),
                                        Ty::new_bool(self.interner()),
                                    )),
                                    _ => None,
                                };
                                o.ok_or(MirLowerError::UnresolvedField)
                            })
                            .collect::<Result<'_, _>>()?,
                    ),
                    expr_id.into(),
                );
                Ok(Some(current))
            }
            Expr::Closure { .. } => {
                let ty = self.expr_ty_without_adjust(expr_id);
                let TyKind::Closure(id, _) = ty.kind() else {
                    not_supported!("closure with non closure type");
                };
                self.result.closures.push(id.0);
                let (captures, _) = self.infer.closure_info(id.0);
                let mut operands = vec![];
                for capture in captures.iter() {
                    let p = Place {
                        local: self.binding_local(capture.place.local)?,
                        projection: self.result.projection_store.intern(
                            capture
                                .place
                                .projections
                                .clone()
                                .into_iter()
                                .map(|it| match it {
                                    HirPlaceProjection::Deref => ProjectionElem::Deref,
                                    HirPlaceProjection::Field(field_id) => {
                                        ProjectionElem::Field(Either::Left(field_id))
                                    }
                                    HirPlaceProjection::TupleField(idx) => {
                                        ProjectionElem::Field(Either::Right(TupleFieldId {
                                            tuple: TupleId(!0), // Dummy as it's unused
                                            index: idx,
                                        }))
                                    }
                                })
                                .collect(),
                        ),
                    };
                    match &capture.kind {
                        CaptureKind::ByRef(bk) => {
                            let tmp_ty = capture.ty.get().instantiate_identity();
                            // FIXME: Handle more than one span.
                            let capture_spans = capture.spans();
                            let tmp: Place = self.temp(tmp_ty, current, capture_spans[0])?.into();
                            self.push_assignment(
                                current,
                                tmp,
                                Rvalue::Ref(*bk, p),
                                capture_spans[0],
                            );
                            operands.push(Operand { kind: OperandKind::Move(tmp), span: None });
                        }
                        CaptureKind::ByValue => {
                            operands.push(Operand { kind: OperandKind::Move(p), span: None })
                        }
                    }
                }
                self.push_assignment(
                    current,
                    place,
                    Rvalue::Aggregate(AggregateKind::Closure(ty.store()), operands.into()),
                    expr_id.into(),
                );
                Ok(Some(current))
            }
            Expr::Tuple { exprs } => {
                let Some(values) = exprs
                    .iter()
                    .map(|it| {
                        let Some((o, c)) = self.lower_expr_to_some_operand(*it, current)? else {
                            return Ok(None);
                        };
                        current = c;
                        Ok(Some(o))
                    })
                    .collect::<Result<'_, Option<_>>>()?
                else {
                    return Ok(None);
                };
                let r = Rvalue::Aggregate(
                    AggregateKind::Tuple(self.expr_ty_without_adjust(expr_id).store()),
                    values,
                );
                self.push_assignment(current, place, r, expr_id.into());
                Ok(Some(current))
            }
            Expr::Array(l) => match l {
                Array::ElementList { elements, .. } => {
                    let elem_ty = match self.expr_ty_without_adjust(expr_id).kind() {
                        TyKind::Array(ty, _) => ty,
                        _ => {
                            return Err(MirLowerError::TypeError(
                                "Array expression with non array type",
                            ));
                        }
                    };
                    let Some(values) = elements
                        .iter()
                        .map(|it| {
                            let Some((o, c)) = self.lower_expr_to_some_operand(*it, current)?
                            else {
                                return Ok(None);
                            };
                            current = c;
                            Ok(Some(o))
                        })
                        .collect::<Result<'_, Option<_>>>()?
                    else {
                        return Ok(None);
                    };
                    let r = Rvalue::Aggregate(AggregateKind::Array(elem_ty.store()), values);
                    self.push_assignment(current, place, r, expr_id.into());
                    Ok(Some(current))
                }
                Array::Repeat { initializer, .. } => {
                    let Some((init, current)) =
                        self.lower_expr_to_some_operand(*initializer, current)?
                    else {
                        return Ok(None);
                    };
                    let len = match self.expr_ty_without_adjust(expr_id).kind() {
                        TyKind::Array(_, len) => len,
                        _ => {
                            return Err(MirLowerError::TypeError(
                                "Array repeat expression with non array type",
                            ));
                        }
                    };
                    let r = Rvalue::Repeat(init, len.store());
                    self.push_assignment(current, place, r, expr_id.into());
                    Ok(Some(current))
                }
            },
            Expr::Literal(l) => {
                let ty = self.expr_ty_without_adjust(expr_id);
                let op = self.lower_literal_to_operand(ty, l)?;
                self.push_assignment(current, place, op.into(), expr_id.into());
                Ok(Some(current))
            }
            Expr::Underscore => Ok(Some(current)),
        }
    }

    fn push_field_projection(&mut self, place: &mut Place, expr_id: ExprId) -> Result<'db, ()> {
        if let Expr::Field { expr, name } = &self.body[expr_id] {
            if let TyKind::Tuple(..) = self.expr_ty_after_adjustments(*expr).kind() {
                let index =
                    name.as_tuple_index().ok_or(MirLowerError::TypeError("named field on tuple"))?
                        as u32;
                *place = place.project(
                    ProjectionElem::Field(Either::Right(TupleFieldId {
                        tuple: TupleId(!0), // dummy as its unused
                        index,
                    })),
                    &mut self.result.projection_store,
                )
            } else {
                let field =
                    self.infer.field_resolution(expr_id).ok_or(MirLowerError::UnresolvedField)?;
                *place =
                    place.project(ProjectionElem::Field(field), &mut self.result.projection_store);
            }
        } else {
            not_supported!("")
        }
        Ok(())
    }

    fn lower_literal_or_const_to_operand(
        &mut self,
        ty: Ty<'db>,
        loc: &ExprId,
    ) -> Result<'db, Operand> {
        match &self.body[*loc] {
            Expr::Literal(l) => self.lower_literal_to_operand(ty, l),
            Expr::Path(c) => {
                let owner = self.owner;
                let db = self.db;
                let unresolved_name = || {
                    MirLowerError::unresolved_path(
                        self.db,
                        c,
                        DisplayTarget::from_crate(db, owner.krate(db)),
                        self.body,
                    )
                };
                let pr = self
                    .resolver
                    .resolve_path_in_value_ns(self.db, c, HygieneId::ROOT)
                    .ok_or_else(unresolved_name)?;
                match pr {
                    ResolveValueResult::ValueNs(v, _) => {
                        if let ValueNs::ConstId(c) = v {
                            self.lower_const_to_operand(
                                GenericArgs::empty(self.interner()),
                                c.into(),
                            )
                        } else {
                            not_supported!("bad path in range pattern");
                        }
                    }
                    ResolveValueResult::Partial(_, _, _) => {
                        not_supported!("associated constants in range pattern")
                    }
                }
            }
            _ => {
                not_supported!("only `char` and numeric types are allowed in range patterns");
            }
        }
    }

    fn lower_literal_to_operand(&mut self, ty: Ty<'db>, l: &Literal) -> Result<'db, Operand> {
        let size = || {
            self.db
                .layout_of_ty(
                    ty.store(),
                    ParamEnvAndCrate { param_env: self.env, krate: self.krate() }.store(),
                )
                .map(|it| it.size.bytes_usize())
        };
        const USIZE_SIZE: usize = size_of::<usize>();
        let bytes: Box<[_]> = match l {
            hir_def::hir::Literal::String(b) => {
                let b = b.as_str();
                let mut data = [0; { 2 * USIZE_SIZE }];
                data[..USIZE_SIZE].copy_from_slice(&0usize.to_le_bytes());
                data[USIZE_SIZE..].copy_from_slice(&b.len().to_le_bytes());
                let mm = MemoryMap::simple(b.as_bytes().into());
                return Ok(Operand::from_concrete_const(Box::new(data), mm, ty));
            }
            hir_def::hir::Literal::CString(b) => {
                let bytes = b.iter().copied().chain(iter::once(0)).collect::<Box<_>>();

                let mut data = [0; { 2 * USIZE_SIZE }];
                data[..USIZE_SIZE].copy_from_slice(&0usize.to_le_bytes());
                data[USIZE_SIZE..].copy_from_slice(&bytes.len().to_le_bytes());
                let mm = MemoryMap::simple(bytes);
                return Ok(Operand::from_concrete_const(Box::new(data), mm, ty));
            }
            hir_def::hir::Literal::ByteString(b) => {
                let mut data = [0; { 2 * USIZE_SIZE }];
                data[..USIZE_SIZE].copy_from_slice(&0usize.to_le_bytes());
                data[USIZE_SIZE..].copy_from_slice(&b.len().to_le_bytes());
                let mm = MemoryMap::simple(b.clone());
                return Ok(Operand::from_concrete_const(Box::new(data), mm, ty));
            }
            hir_def::hir::Literal::Char(c) => Box::new(u32::from(*c).to_le_bytes()),
            hir_def::hir::Literal::Bool(b) => Box::new([*b as u8]),
            hir_def::hir::Literal::Int(it, _) => Box::from(&it.to_le_bytes()[0..size()?]),
            hir_def::hir::Literal::Uint(it, _) => Box::from(&it.to_le_bytes()[0..size()?]),
            hir_def::hir::Literal::Float(f, _) => match size()? {
                16 => Box::new(f.to_f128().to_bits().to_le_bytes()),
                8 => Box::new(f.to_f64().to_le_bytes()),
                4 => Box::new(f.to_f32().to_le_bytes()),
                2 => Box::new(u16::try_from(f.to_f16().to_bits()).unwrap().to_le_bytes()),
                _ => {
                    return Err(MirLowerError::TypeError(
                        "float with size other than 2, 4, 8 or 16 bytes",
                    ));
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
        const_id: GeneralConstId,
        prev_block: BasicBlockId,
        place: Place,
        subst: GenericArgs<'db>,
        span: MirSpan,
    ) -> Result<'db, ()> {
        let c = self.lower_const_to_operand(subst, const_id)?;
        self.push_assignment(prev_block, place, c.into(), span);
        Ok(())
    }

    fn lower_const_to_operand(
        &mut self,
        subst: GenericArgs<'db>,
        const_id: GeneralConstId,
    ) -> Result<'db, Operand> {
        let konst = if !subst.is_empty() {
            // We can't evaluate constant with substitution now, as generics are not monomorphized in lowering.
            Const::new_unevaluated(
                self.interner(),
                UnevaluatedConst { def: const_id.into(), args: subst },
            )
        } else {
            match const_id {
                id @ GeneralConstId::ConstId(const_id) => {
                    self.db.const_eval(const_id, subst, None).map_err(|e| {
                        let name = id.name(self.db);
                        MirLowerError::ConstEvalError(name.into(), Box::new(e))
                    })?
                }
                GeneralConstId::StaticId(static_id) => {
                    self.db.const_eval_static(static_id).map_err(|e| {
                        let name = const_id.name(self.db);
                        MirLowerError::ConstEvalError(name.into(), Box::new(e))
                    })?
                }
            }
        };
        let ty = self
            .db
            .value_ty(match const_id {
                GeneralConstId::ConstId(id) => id.into(),
                GeneralConstId::StaticId(id) => id.into(),
            })
            .unwrap()
            .instantiate(self.interner(), subst);
        Ok(Operand {
            kind: OperandKind::Constant { konst: konst.store(), ty: ty.store() },
            span: None,
        })
    }

    fn write_bytes_to_place(
        &mut self,
        prev_block: BasicBlockId,
        place: Place,
        cv: Box<[u8]>,
        ty: Ty<'db>,
        span: MirSpan,
    ) -> Result<'db, ()> {
        self.push_assignment(prev_block, place, Operand::from_bytes(cv, ty).into(), span);
        Ok(())
    }

    fn lower_enum_variant(
        &mut self,
        variant_id: EnumVariantId,
        prev_block: BasicBlockId,
        place: Place,
        ty: Ty<'db>,
        fields: Box<[Operand]>,
        span: MirSpan,
    ) -> Result<'db, BasicBlockId> {
        let subst = match ty.kind() {
            TyKind::Adt(_, subst) => subst,
            _ => implementation_error!("Non ADT enum"),
        };
        self.push_assignment(
            prev_block,
            place,
            Rvalue::Aggregate(AggregateKind::Adt(variant_id.into(), subst.store()), fields),
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
        span: MirSpan,
    ) -> Result<'db, Option<BasicBlockId>> {
        let Some(args) = args
            .map(|arg| {
                if let Some((temp, c)) = self.lower_expr_to_some_operand(arg, current)? {
                    current = c;
                    Ok(Some(temp))
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<'_, Option<Vec<_>>>>()?
        else {
            return Ok(None);
        };
        self.lower_call(func, args.into(), place, current, is_uninhabited, span)
    }

    fn lower_call(
        &mut self,
        func: Operand,
        args: Box<[Operand]>,
        place: Place,
        current: BasicBlockId,
        is_uninhabited: bool,
        span: MirSpan,
    ) -> Result<'db, Option<BasicBlockId>> {
        let b = if is_uninhabited { None } else { Some(self.new_basic_block()) };
        self.set_terminator(
            current,
            TerminatorKind::Call {
                func,
                args,
                destination: place,
                target: b,
                cleanup: None,
                from_hir_call: true,
            },
            span,
        );
        Ok(b)
    }

    fn is_unterminated(&mut self, source: BasicBlockId) -> bool {
        self.result.basic_blocks[source].terminator.is_none()
    }

    fn set_terminator(&mut self, source: BasicBlockId, terminator: TerminatorKind, span: MirSpan) {
        self.result.basic_blocks[source].terminator = Some(Terminator { span, kind: terminator });
    }

    fn set_goto(&mut self, source: BasicBlockId, target: BasicBlockId, span: MirSpan) {
        self.set_terminator(source, TerminatorKind::Goto { target }, span);
    }

    fn expr_ty_without_adjust(&self, e: ExprId) -> Ty<'db> {
        self.infer.expr_ty(e)
    }

    fn expr_ty_after_adjustments(&self, e: ExprId) -> Ty<'db> {
        let mut ty = None;
        if let Some(it) = self.infer.expr_adjustments.get(&e)
            && let Some(it) = it.last()
        {
            ty = Some(it.target.as_ref());
        }
        ty.unwrap_or_else(|| self.expr_ty_without_adjust(e))
    }

    fn push_statement(&mut self, block: BasicBlockId, statement: Statement) {
        self.result.basic_blocks[block].statements.push(statement);
    }

    fn push_fake_read(&mut self, block: BasicBlockId, p: Place, span: MirSpan) {
        self.push_statement(block, StatementKind::FakeRead(p).with_span(span));
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

    fn discr_temp_place(&mut self, current: BasicBlockId) -> Place {
        match &self.discr_temp {
            Some(it) => *it,
            None => {
                // FIXME: rustc's ty is dependent on the adt type, maybe we need to do that as well
                let discr_ty = Ty::new_int(self.interner(), rustc_type_ir::IntTy::I128);
                let tmp: Place = self
                    .temp(discr_ty, current, MirSpan::Unknown)
                    .expect("discr_ty is never unsized")
                    .into();
                self.discr_temp = Some(tmp);
                tmp
            }
        }
    }

    fn lower_loop(
        &mut self,
        prev_block: BasicBlockId,
        place: Place,
        label: Option<LabelId>,
        span: MirSpan,
        f: impl FnOnce(&mut MirLowerCtx<'_, 'db>, BasicBlockId) -> Result<'db, ()>,
    ) -> Result<'db, Option<BasicBlockId>> {
        let begin = self.new_basic_block();
        let prev = self.current_loop_blocks.replace(LoopBlocks {
            begin,
            end: None,
            place,
            drop_scope_index: self.drop_scopes.len(),
        });
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
        self.set_goto(prev_block, begin, span);
        f(self, begin)?;
        let my = mem::replace(&mut self.current_loop_blocks, prev).ok_or(
            MirLowerError::ImplementationError("current_loop_blocks is corrupt".to_owned()),
        )?;
        if let Some(prev) = prev_label {
            self.labeled_loop_blocks.insert(label.unwrap(), prev);
        }
        Ok(my.end)
    }

    fn has_adjustments(&self, expr_id: ExprId) -> bool {
        !self.infer.expr_adjustments.get(&expr_id).map(|it| it.is_empty()).unwrap_or(true)
    }

    fn merge_blocks(
        &mut self,
        b1: Option<BasicBlockId>,
        b2: Option<BasicBlockId>,
        span: MirSpan,
    ) -> Option<BasicBlockId> {
        match (b1, b2) {
            (None, None) => None,
            (None, Some(b)) | (Some(b), None) => Some(b),
            (Some(b1), Some(b2)) => {
                let bm = self.new_basic_block();
                self.set_goto(b1, bm, span);
                self.set_goto(b2, bm, span);
                Some(bm)
            }
        }
    }

    fn current_loop_end(&mut self) -> Result<'db, BasicBlockId> {
        let r = match self
            .current_loop_blocks
            .as_mut()
            .ok_or(MirLowerError::ImplementationError(
                "Current loop access out of loop".to_owned(),
            ))?
            .end
        {
            Some(it) => it,
            None => {
                let s = self.new_basic_block();
                self.current_loop_blocks
                    .as_mut()
                    .ok_or(MirLowerError::ImplementationError(
                        "Current loop access out of loop".to_owned(),
                    ))?
                    .end = Some(s);
                s
            }
        };
        Ok(r)
    }

    fn is_uninhabited(&self, expr_id: ExprId) -> bool {
        is_ty_uninhabited_from(
            &self.infcx,
            self.infer.expr_ty(expr_id),
            self.owner.module(self.db),
            self.env,
        )
    }

    /// This function push `StorageLive` statement for the binding, and applies changes to add `StorageDead` and
    /// `Drop` in the appropriated places.
    fn push_storage_live(&mut self, b: BindingId, current: BasicBlockId) -> Result<'db, ()> {
        let l = self.binding_local(b)?;
        self.push_storage_live_for_local(l, current, MirSpan::BindingId(b))
    }

    fn push_storage_live_for_local(
        &mut self,
        l: LocalId,
        current: BasicBlockId,
        span: MirSpan,
    ) -> Result<'db, ()> {
        self.drop_scopes.last_mut().unwrap().locals.push(l);
        self.push_statement(current, StatementKind::StorageLive(l).with_span(span));
        Ok(())
    }

    fn lower_block_to_place(
        &mut self,
        statements: &[hir_def::hir::Statement],
        mut current: BasicBlockId,
        tail: Option<ExprId>,
        place: Place,
        span: MirSpan,
    ) -> Result<'db, Option<Idx<BasicBlock>>> {
        let scope = self.push_drop_scope();
        for statement in statements.iter() {
            match statement {
                hir_def::hir::Statement::Let { pat, initializer, else_branch, type_ref: _ } => {
                    if let Some(expr_id) = initializer {
                        let else_block;
                        let Some((init_place, c)) =
                            self.lower_expr_as_place(current, *expr_id, true)?
                        else {
                            scope.pop_assume_dropped(self);
                            return Ok(None);
                        };
                        current = c;
                        self.push_fake_read(current, init_place, span);
                        // Using the initializer for the resolver scope is good enough for us, as it cannot create new declarations
                        // and has all declarations of the `let`.
                        let resolver_guard =
                            self.resolver.update_to_inner_scope(self.db, self.owner, *expr_id);
                        (current, else_block) =
                            self.pattern_match(current, None, init_place, *pat)?;
                        self.resolver.reset_to_guard(resolver_guard);
                        match (else_block, else_branch) {
                            (None, _) => (),
                            (Some(else_block), None) => {
                                self.set_terminator(else_block, TerminatorKind::Unreachable, span);
                            }
                            (Some(else_block), Some(else_branch)) => {
                                if let Some((_, b)) =
                                    self.lower_expr_as_place(else_block, *else_branch, true)?
                                {
                                    self.set_terminator(b, TerminatorKind::Unreachable, span);
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
                &hir_def::hir::Statement::Expr { expr, has_semi: _ } => {
                    let scope2 = self.push_drop_scope();
                    let Some((p, c)) = self.lower_expr_as_place(current, expr, true)? else {
                        scope2.pop_assume_dropped(self);
                        scope.pop_assume_dropped(self);
                        return Ok(None);
                    };
                    self.push_fake_read(c, p, expr.into());
                    current = scope2.pop_and_drop(self, c, expr.into());
                }
                hir_def::hir::Statement::Item(_) => (),
            }
        }
        if let Some(tail) = tail {
            let Some(c) = self.lower_expr_to_place(tail, place, current)? else {
                scope.pop_assume_dropped(self);
                return Ok(None);
            };
            current = c;
        }
        current = scope.pop_and_drop(self, current, span);
        Ok(Some(current))
    }

    fn lower_params_and_bindings(
        &mut self,
        params: impl Iterator<Item = (PatId, Ty<'db>)> + Clone,
        self_binding: Option<(BindingId, Ty<'db>)>,
        pick_binding: impl Fn(BindingId) -> bool,
    ) -> Result<'db, BasicBlockId> {
        let base_param_count = self.result.param_locals.len();
        let self_binding = match self_binding {
            Some((self_binding, ty)) => {
                let local_id = self.result.locals.alloc(Local { ty: ty.store() });
                self.drop_scopes.last_mut().unwrap().locals.push(local_id);
                self.result.binding_locals.insert(self_binding, local_id);
                self.result.param_locals.push(local_id);
                Some(self_binding)
            }
            None => None,
        };
        self.result.param_locals.extend(params.clone().map(|(it, ty)| {
            let local_id = self.result.locals.alloc(Local { ty: ty.store() });
            self.drop_scopes.last_mut().unwrap().locals.push(local_id);
            if let Pat::Bind { id, subpat: None } = self.body[it]
                && matches!(
                    self.body[id].mode,
                    BindingAnnotation::Unannotated | BindingAnnotation::Mutable
                )
            {
                self.result.binding_locals.insert(id, local_id);
            }
            local_id
        }));
        // and then rest of bindings
        for (id, _) in self.body.bindings() {
            if !pick_binding(id) {
                continue;
            }
            if !self.result.binding_locals.contains_idx(id) {
                self.result.binding_locals.insert(
                    id,
                    self.result.locals.alloc(Local { ty: self.infer.binding_ty(id).store() }),
                );
            }
        }
        let mut current = self.result.start_block;
        if let Some(self_binding) = self_binding {
            let local = self.result.param_locals.clone()[base_param_count];
            if local != self.binding_local(self_binding)? {
                let r = self.match_self_param(self_binding, current, local)?;
                if let Some(b) = r.1 {
                    self.set_terminator(b, TerminatorKind::Unreachable, MirSpan::SelfParam);
                }
                current = r.0;
            }
        }
        let local_params = self
            .result
            .param_locals
            .clone()
            .into_iter()
            .skip(base_param_count + self_binding.is_some() as usize);
        for ((param, _), local) in params.zip(local_params) {
            if let Pat::Bind { id, .. } = self.body[param]
                && local == self.binding_local(id)?
            {
                continue;
            }
            let r = self.pattern_match(current, None, local.into(), param)?;
            if let Some(b) = r.1 {
                self.set_terminator(b, TerminatorKind::Unreachable, param.into());
            }
            current = r.0;
        }
        Ok(current)
    }

    fn binding_local(&self, b: BindingId) -> Result<'db, LocalId> {
        match self.result.binding_locals.get(b) {
            Some(it) => Ok(*it),
            None => {
                // FIXME: It should never happens, but currently it will happen in `const_dependent_on_local` test, which
                // is a hir lowering problem IMO.
                // never!("Using inaccessible local for binding is always a bug");
                Err(MirLowerError::InaccessibleLocal)
            }
        }
    }

    fn const_eval_discriminant(&self, variant: EnumVariantId) -> Result<'db, i128> {
        let r = self.db.const_eval_discriminant(variant);
        match r {
            Ok(r) => Ok(r),
            Err(e) => {
                let edition = self.edition();
                let db = self.db;
                let loc = variant.lookup(db);
                let name = format!(
                    "{}::{}",
                    self.db.enum_signature(loc.parent).name.display(db, edition),
                    loc.parent
                        .enum_variants(self.db)
                        .variant_name_by_id(variant)
                        .unwrap()
                        .display(db, edition),
                );
                Err(MirLowerError::ConstEvalError(name.into(), Box::new(e)))
            }
        }
    }

    fn edition(&self) -> Edition {
        self.krate().data(self.db).edition
    }

    fn krate(&self) -> Crate {
        self.owner.krate(self.db)
    }

    fn display_target(&self) -> DisplayTarget {
        DisplayTarget::from_crate(self.db, self.krate())
    }

    fn drop_until_scope(
        &mut self,
        scope_index: usize,
        mut current: BasicBlockId,
        span: MirSpan,
    ) -> BasicBlockId {
        for scope in self.drop_scopes[scope_index..].to_vec().iter().rev() {
            self.emit_drop_and_storage_dead_for_scope(scope, &mut current, span);
        }
        current
    }

    fn push_drop_scope(&mut self) -> DropScopeToken {
        self.drop_scopes.push(DropScope::default());
        DropScopeToken
    }

    /// Don't call directly
    fn pop_drop_scope_assume_dropped_internal(&mut self) {
        self.drop_scopes.pop();
    }

    /// Don't call directly
    fn pop_drop_scope_internal(
        &mut self,
        mut current: BasicBlockId,
        span: MirSpan,
    ) -> BasicBlockId {
        let scope = self.drop_scopes.pop().unwrap();
        self.emit_drop_and_storage_dead_for_scope(&scope, &mut current, span);
        current
    }

    fn pop_drop_scope_assert_finished(
        &mut self,
        mut current: BasicBlockId,
        span: MirSpan,
    ) -> Result<'db, BasicBlockId> {
        current = self.pop_drop_scope_internal(current, span);
        if !self.drop_scopes.is_empty() {
            implementation_error!("Mismatched count between drop scope push and pops");
        }
        Ok(current)
    }

    fn emit_drop_and_storage_dead_for_scope(
        &mut self,
        scope: &DropScope,
        current: &mut Idx<BasicBlock>,
        span: MirSpan,
    ) {
        for &l in scope.locals.iter().rev() {
            if !self.infcx.type_is_copy_modulo_regions(self.env, self.result.locals[l].ty.as_ref())
            {
                let prev = std::mem::replace(current, self.new_basic_block());
                self.set_terminator(
                    prev,
                    TerminatorKind::Drop { place: l.into(), target: *current, unwind: None },
                    span,
                );
            }
            self.push_statement(*current, StatementKind::StorageDead(l).with_span(span));
        }
    }
}

fn cast_kind<'db>(
    db: &'db dyn HirDatabase,
    source_ty: Ty<'db>,
    target_ty: Ty<'db>,
) -> Result<'db, CastKind> {
    let from = CastTy::from_ty(db, source_ty);
    let cast = CastTy::from_ty(db, target_ty);
    Ok(match (from, cast) {
        (Some(CastTy::Ptr(..) | CastTy::FnPtr), Some(CastTy::Int(_))) => {
            CastKind::PointerExposeAddress
        }
        (Some(CastTy::Int(_)), Some(CastTy::Ptr(..))) => CastKind::PointerFromExposedAddress,
        (Some(CastTy::Int(_)), Some(CastTy::Int(_))) => CastKind::IntToInt,
        (Some(CastTy::FnPtr), Some(CastTy::Ptr(..))) => CastKind::FnPtrToPtr,
        (Some(CastTy::Float), Some(CastTy::Int(_))) => CastKind::FloatToInt,
        (Some(CastTy::Int(_)), Some(CastTy::Float)) => CastKind::IntToFloat,
        (Some(CastTy::Float), Some(CastTy::Float)) => CastKind::FloatToFloat,
        (Some(CastTy::Ptr(..)), Some(CastTy::Ptr(..))) => CastKind::PtrToPtr,
        _ => not_supported!("Unknown cast between {source_ty:?} and {target_ty:?}"),
    })
}

pub fn mir_body_for_closure_query<'db>(
    db: &'db dyn HirDatabase,
    closure: InternedClosureId,
) -> Result<'db, Arc<MirBody>> {
    let InternedClosure(owner, expr) = db.lookup_intern_closure(closure);
    let body = db.body(owner);
    let infer = InferenceResult::for_body(db, owner);
    let Expr::Closure { args, body: root, .. } = &body[expr] else {
        implementation_error!("closure expression is not closure");
    };
    let crate::next_solver::TyKind::Closure(_, substs) = infer.expr_ty(expr).kind() else {
        implementation_error!("closure expression is not closure");
    };
    let (captures, kind) = infer.closure_info(closure);
    let mut ctx = MirLowerCtx::new(db, owner, &body, infer);
    // 0 is return local
    ctx.result.locals.alloc(Local { ty: infer.expr_ty(*root).store() });
    let closure_local = ctx.result.locals.alloc(Local {
        ty: match kind {
            FnTrait::FnOnce | FnTrait::AsyncFnOnce => infer.expr_ty(expr),
            FnTrait::FnMut | FnTrait::AsyncFnMut => Ty::new_ref(
                ctx.interner(),
                Region::error(ctx.interner()),
                infer.expr_ty(expr),
                Mutability::Mut,
            ),
            FnTrait::Fn | FnTrait::AsyncFn => Ty::new_ref(
                ctx.interner(),
                Region::error(ctx.interner()),
                infer.expr_ty(expr),
                Mutability::Not,
            ),
        }
        .store(),
    });
    ctx.result.param_locals.push(closure_local);
    let sig = ctx.interner().signature_unclosure(substs.as_closure().sig(), Safety::Safe);
    let resolver_guard = ctx.resolver.update_to_inner_scope(db, owner, expr);
    let current = ctx.lower_params_and_bindings(
        args.iter().zip(sig.skip_binder().inputs().iter()).map(|(it, y)| (*it, *y)),
        None,
        |_| true,
    )?;
    ctx.resolver.reset_to_guard(resolver_guard);
    if let Some(current) = ctx.lower_expr_to_place(*root, return_slot().into(), current)? {
        let current = ctx.pop_drop_scope_assert_finished(current, root.into())?;
        ctx.set_terminator(current, TerminatorKind::Return, (*root).into());
    }
    let mut upvar_map: FxHashMap<LocalId, Vec<(&CapturedItem, usize)>> = FxHashMap::default();
    for (i, capture) in captures.iter().enumerate() {
        let local = ctx.binding_local(capture.place.local)?;
        upvar_map.entry(local).or_default().push((capture, i));
    }
    let mut err = None;
    let closure_local = ctx.result.locals.iter().nth(1).unwrap().0;
    let closure_projection = match kind {
        FnTrait::FnOnce | FnTrait::AsyncFnOnce => vec![],
        FnTrait::FnMut | FnTrait::Fn | FnTrait::AsyncFnMut | FnTrait::AsyncFn => {
            vec![ProjectionElem::Deref]
        }
    };
    ctx.result.walk_places(|p, store| {
        if let Some(it) = upvar_map.get(&p.local) {
            let r = it.iter().find(|it| {
                if p.projection.lookup(store).len() < it.0.place.projections.len() {
                    return false;
                }
                for (it, y) in p.projection.lookup(store).iter().zip(it.0.place.projections.iter())
                {
                    match (it, y) {
                        (ProjectionElem::Deref, HirPlaceProjection::Deref) => (),
                        (ProjectionElem::Field(Either::Left(it)), HirPlaceProjection::Field(y))
                            if it == y => {}
                        (
                            ProjectionElem::Field(Either::Right(it)),
                            HirPlaceProjection::TupleField(y),
                        ) if it.index == *y => (),
                        _ => return false,
                    }
                }
                true
            });
            match r {
                Some(it) => {
                    p.local = closure_local;
                    let mut next_projs = closure_projection.clone();
                    next_projs.push(PlaceElem::ClosureField(it.1));
                    let prev_projs = p.projection;
                    if it.0.kind != CaptureKind::ByValue {
                        next_projs.push(ProjectionElem::Deref);
                    }
                    next_projs.extend(
                        prev_projs.lookup(store).iter().skip(it.0.place.projections.len()).cloned(),
                    );
                    p.projection = store.intern(next_projs.into());
                }
                None => err = Some(*p),
            }
        }
    });
    ctx.result.binding_locals = ctx
        .result
        .binding_locals
        .into_iter()
        .filter(|it| ctx.body.binding_owner(it.0) == Some(expr))
        .collect();
    if let Some(err) = err {
        return Err(MirLowerError::UnresolvedUpvar(err));
    }
    ctx.result.shrink_to_fit();
    Ok(Arc::new(ctx.result))
}

pub fn mir_body_query<'db>(
    db: &'db dyn HirDatabase,
    def: DefWithBodyId,
) -> Result<'db, Arc<MirBody>> {
    let krate = def.krate(db);
    let edition = krate.data(db).edition;
    let detail = match def {
        DefWithBodyId::FunctionId(it) => {
            db.function_signature(it).name.display(db, edition).to_string()
        }
        DefWithBodyId::StaticId(it) => {
            db.static_signature(it).name.display(db, edition).to_string()
        }
        DefWithBodyId::ConstId(it) => db
            .const_signature(it)
            .name
            .clone()
            .unwrap_or_else(Name::missing)
            .display(db, edition)
            .to_string(),
        DefWithBodyId::VariantId(it) => {
            let loc = it.lookup(db);
            loc.parent.enum_variants(db).variants[loc.index as usize]
                .1
                .display(db, edition)
                .to_string()
        }
    };
    let _p = tracing::info_span!("mir_body_query", ?detail).entered();
    let body = db.body(def);
    let infer = InferenceResult::for_body(db, def);
    let mut result = lower_to_mir(db, def, &body, infer, body.body_expr)?;
    result.shrink_to_fit();
    Ok(Arc::new(result))
}

pub(crate) fn mir_body_cycle_result<'db>(
    _db: &'db dyn HirDatabase,
    _: salsa::Id,
    _def: DefWithBodyId,
) -> Result<'db, Arc<MirBody>> {
    Err(MirLowerError::Loop)
}

pub fn lower_to_mir<'db>(
    db: &'db dyn HirDatabase,
    owner: DefWithBodyId,
    body: &Body,
    infer: &InferenceResult,
    // FIXME: root_expr should always be the body.body_expr, but since `X` in `[(); X]` doesn't have its own specific body yet, we
    // need to take this input explicitly.
    root_expr: ExprId,
) -> Result<'db, MirBody> {
    if infer.type_mismatches().next().is_some() || infer.is_erroneous() {
        return Err(MirLowerError::HasErrors);
    }
    let mut ctx = MirLowerCtx::new(db, owner, body, infer);
    // 0 is return local
    ctx.result.locals.alloc(Local { ty: ctx.expr_ty_after_adjustments(root_expr).store() });
    let binding_picker = |b: BindingId| {
        let owner = ctx.body.binding_owner(b);
        if root_expr == body.body_expr { owner.is_none() } else { owner == Some(root_expr) }
    };
    // 1 to param_len is for params
    // FIXME: replace with let chain once it becomes stable
    let current = 'b: {
        if body.body_expr == root_expr {
            // otherwise it's an inline const, and has no parameter
            if let DefWithBodyId::FunctionId(fid) = owner {
                let callable_sig =
                    db.callable_item_signature(fid.into()).instantiate_identity().skip_binder();
                let mut params = callable_sig.inputs().iter().copied();
                let self_param = body.self_param.and_then(|id| Some((id, params.next()?)));
                break 'b ctx.lower_params_and_bindings(
                    body.params.iter().zip(params).map(|(it, y)| (*it, y)),
                    self_param,
                    binding_picker,
                )?;
            }
        }
        ctx.lower_params_and_bindings([].into_iter(), None, binding_picker)?
    };
    if let Some(current) = ctx.lower_expr_to_place(root_expr, return_slot().into(), current)? {
        let current = ctx.pop_drop_scope_assert_finished(current, root_expr.into())?;
        ctx.set_terminator(current, TerminatorKind::Return, root_expr.into());
    }
    Ok(ctx.result)
}
