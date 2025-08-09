//! Lookup hir elements using positions in the source code. This is a lossy
//! transformation: in general, a single source might correspond to several
//! modules, functions, etc, due to macros, cfgs and `#[path=]` attributes on
//! modules.
//!
//! So, this modules should not be used during hir construction, it exists
//! purely for "IDE needs".
use std::iter::{self, once};

use crate::{
    Adt, AssocItem, BindingMode, BuiltinAttr, BuiltinType, Callable, Const, DeriveHelper, Field,
    Function, GenericSubstitution, Local, Macro, ModuleDef, Static, Struct, ToolModule, Trait,
    TraitAlias, TupleField, Type, TypeAlias, Variant,
    db::HirDatabase,
    semantics::{PathResolution, PathResolutionPerNs},
};
use either::Either;
use hir_def::{
    AdtId, AssocItemId, CallableDefId, ConstId, DefWithBodyId, FieldId, FunctionId, GenericDefId,
    ItemContainerId, LocalFieldId, Lookup, ModuleDefId, StructId, TraitId, VariantId,
    expr_store::{
        Body, BodySourceMap, ExpressionStore, ExpressionStoreSourceMap, HygieneId,
        lower::ExprCollector,
        path::Path,
        scope::{ExprScopes, ScopeId},
    },
    hir::{BindingId, Expr, ExprId, ExprOrPatId, Pat},
    lang_item::LangItem,
    nameres::MacroSubNs,
    resolver::{HasResolver, Resolver, TypeNs, ValueNs, resolver_for_scope},
    type_ref::{Mutability, TypeRefId},
};
use hir_expand::{
    HirFileId, InFile,
    mod_path::{ModPath, PathKind, path},
    name::{AsName, Name},
};
use hir_ty::{
    Adjustment, AliasTy, InferenceResult, Interner, LifetimeElisionKind, ProjectionTy,
    Substitution, ToChalk, TraitEnvironment, Ty, TyExt, TyKind, TyLoweringContext,
    diagnostics::{
        InsideUnsafeBlock, record_literal_missing_fields, record_pattern_missing_fields,
        unsafe_operations,
    },
    from_assoc_type_id,
    lang_items::lang_items_for_bin_op,
    method_resolution,
};
use intern::sym;
use itertools::Itertools;
use smallvec::SmallVec;
use stdx::never;
use syntax::{
    SyntaxKind, SyntaxNode, TextRange, TextSize,
    ast::{self, AstNode, RangeItem, RangeOp},
};
use triomphe::Arc;

/// `SourceAnalyzer` is a convenience wrapper which exposes HIR API in terms of
/// original source files. It should not be used inside the HIR itself.
#[derive(Debug)]
pub(crate) struct SourceAnalyzer<'db> {
    pub(crate) file_id: HirFileId,
    pub(crate) resolver: Resolver<'db>,
    pub(crate) body_or_sig: Option<BodyOrSig>,
}

#[derive(Debug)]
pub(crate) enum BodyOrSig {
    Body {
        def: DefWithBodyId,
        body: Arc<Body>,
        source_map: Arc<BodySourceMap>,
        infer: Option<Arc<InferenceResult>>,
    },
    // To be folded into body once it is considered one
    VariantFields {
        def: VariantId,
        store: Arc<ExpressionStore>,
        source_map: Arc<ExpressionStoreSourceMap>,
    },
    Sig {
        def: GenericDefId,
        store: Arc<ExpressionStore>,
        source_map: Arc<ExpressionStoreSourceMap>,
        // infer: Option<Arc<InferenceResult>>,
    },
}

impl<'db> SourceAnalyzer<'db> {
    pub(crate) fn new_for_body(
        db: &'db dyn HirDatabase,
        def: DefWithBodyId,
        node: InFile<&SyntaxNode>,
        offset: Option<TextSize>,
    ) -> SourceAnalyzer<'db> {
        Self::new_for_body_(db, def, node, offset, Some(db.infer(def)))
    }

    pub(crate) fn new_for_body_no_infer(
        db: &'db dyn HirDatabase,
        def: DefWithBodyId,
        node: InFile<&SyntaxNode>,
        offset: Option<TextSize>,
    ) -> SourceAnalyzer<'db> {
        Self::new_for_body_(db, def, node, offset, None)
    }

    pub(crate) fn new_for_body_(
        db: &'db dyn HirDatabase,
        def: DefWithBodyId,
        node @ InFile { file_id, .. }: InFile<&SyntaxNode>,
        offset: Option<TextSize>,
        infer: Option<Arc<InferenceResult>>,
    ) -> SourceAnalyzer<'db> {
        let (body, source_map) = db.body_with_source_map(def);
        let scopes = db.expr_scopes(def);
        let scope = match offset {
            None => scope_for(db, &scopes, &source_map, node),
            Some(offset) => {
                debug_assert!(
                    node.text_range().contains_inclusive(offset),
                    "{:?} not in {:?}",
                    offset,
                    node.text_range()
                );
                scope_for_offset(db, &scopes, &source_map, node.file_id, offset)
            }
        };
        let resolver = resolver_for_scope(db, def, scope);
        SourceAnalyzer {
            resolver,
            body_or_sig: Some(BodyOrSig::Body { def, body, source_map, infer }),
            file_id,
        }
    }

    pub(crate) fn new_generic_def(
        db: &'db dyn HirDatabase,
        def: GenericDefId,
        InFile { file_id, .. }: InFile<&SyntaxNode>,
        _offset: Option<TextSize>,
    ) -> SourceAnalyzer<'db> {
        let (_params, store, source_map) = db.generic_params_and_store_and_source_map(def);
        let resolver = def.resolver(db);
        SourceAnalyzer {
            resolver,
            body_or_sig: Some(BodyOrSig::Sig { def, store, source_map }),
            file_id,
        }
    }

    pub(crate) fn new_variant_body(
        db: &'db dyn HirDatabase,
        def: VariantId,
        InFile { file_id, .. }: InFile<&SyntaxNode>,
        _offset: Option<TextSize>,
    ) -> SourceAnalyzer<'db> {
        let (fields, source_map) = def.fields_with_source_map(db);
        let resolver = def.resolver(db);
        SourceAnalyzer {
            resolver,
            body_or_sig: Some(BodyOrSig::VariantFields {
                def,
                store: fields.store.clone(),
                source_map: source_map.clone(),
            }),
            file_id,
        }
    }

    pub(crate) fn new_for_resolver(
        resolver: Resolver<'db>,
        node: InFile<&SyntaxNode>,
    ) -> SourceAnalyzer<'db> {
        SourceAnalyzer { resolver, body_or_sig: None, file_id: node.file_id }
    }

    // FIXME: Remove this
    fn body_(&self) -> Option<(DefWithBodyId, &Body, &BodySourceMap, Option<&InferenceResult>)> {
        self.body_or_sig.as_ref().and_then(|it| match it {
            BodyOrSig::Body { def, body, source_map, infer } => {
                Some((*def, &**body, &**source_map, infer.as_deref()))
            }
            _ => None,
        })
    }

    fn infer(&self) -> Option<&InferenceResult> {
        self.body_or_sig.as_ref().and_then(|it| match it {
            BodyOrSig::Sig { .. } => None,
            BodyOrSig::VariantFields { .. } => None,
            BodyOrSig::Body { infer, .. } => infer.as_deref(),
        })
    }

    fn body(&self) -> Option<&Body> {
        self.body_or_sig.as_ref().and_then(|it| match it {
            BodyOrSig::Sig { .. } => None,
            BodyOrSig::VariantFields { .. } => None,
            BodyOrSig::Body { body, .. } => Some(&**body),
        })
    }

    pub(crate) fn store(&self) -> Option<&ExpressionStore> {
        self.body_or_sig.as_ref().map(|it| match it {
            BodyOrSig::Sig { store, .. } => &**store,
            BodyOrSig::VariantFields { store, .. } => &**store,
            BodyOrSig::Body { body, .. } => &body.store,
        })
    }

    pub(crate) fn store_sm(&self) -> Option<&ExpressionStoreSourceMap> {
        self.body_or_sig.as_ref().map(|it| match it {
            BodyOrSig::Sig { source_map, .. } => &**source_map,
            BodyOrSig::VariantFields { source_map, .. } => &**source_map,
            BodyOrSig::Body { source_map, .. } => &source_map.store,
        })
    }

    fn trait_environment(&self, db: &'db dyn HirDatabase) -> Arc<TraitEnvironment> {
        self.body_().map(|(def, ..)| def).map_or_else(
            || TraitEnvironment::empty(self.resolver.krate()),
            |def| db.trait_environment_for_body(def),
        )
    }

    fn expr_id(&self, expr: ast::Expr) -> Option<ExprOrPatId> {
        let src = InFile { file_id: self.file_id, value: expr };
        self.store_sm()?.node_expr(src.as_ref())
    }

    fn pat_id(&self, pat: &ast::Pat) -> Option<ExprOrPatId> {
        let src = InFile { file_id: self.file_id, value: pat };
        self.store_sm()?.node_pat(src)
    }

    fn type_id(&self, pat: &ast::Type) -> Option<TypeRefId> {
        let src = InFile { file_id: self.file_id, value: pat };
        self.store_sm()?.node_type(src)
    }

    fn binding_id_of_pat(&self, pat: &ast::IdentPat) -> Option<BindingId> {
        let pat_id = self.pat_id(&pat.clone().into())?;
        if let Pat::Bind { id, .. } = self.store()?[pat_id.as_pat()?] { Some(id) } else { None }
    }

    pub(crate) fn expr_adjustments(&self, expr: &ast::Expr) -> Option<&[Adjustment]> {
        // It is safe to omit destructuring assignments here because they have no adjustments (neither
        // expressions nor patterns).
        let expr_id = self.expr_id(expr.clone())?.as_expr()?;
        let infer = self.infer()?;
        infer.expr_adjustment(expr_id)
    }

    pub(crate) fn type_of_type(
        &self,
        db: &'db dyn HirDatabase,
        ty: &ast::Type,
    ) -> Option<Type<'db>> {
        let type_ref = self.type_id(ty)?;
        let ty = TyLoweringContext::new(
            db,
            &self.resolver,
            self.store()?,
            self.resolver.generic_def()?,
            // FIXME: Is this correct here? Anyway that should impact mostly diagnostics, which we don't emit here
            // (this can impact the lifetimes generated, e.g. in `const` they won't be `'static`, but this seems like a
            // small problem).
            LifetimeElisionKind::Infer,
        )
        .lower_ty(type_ref);
        Some(Type::new_with_resolver(db, &self.resolver, ty))
    }

    pub(crate) fn type_of_expr(
        &self,
        db: &'db dyn HirDatabase,
        expr: &ast::Expr,
    ) -> Option<(Type<'db>, Option<Type<'db>>)> {
        let expr_id = self.expr_id(expr.clone())?;
        let infer = self.infer()?;
        let coerced = expr_id
            .as_expr()
            .and_then(|expr_id| infer.expr_adjustment(expr_id))
            .and_then(|adjusts| adjusts.last().map(|adjust| adjust.target.clone()));
        let ty = infer[expr_id].clone();
        let mk_ty = |ty| Type::new_with_resolver(db, &self.resolver, ty);
        Some((mk_ty(ty), coerced.map(mk_ty)))
    }

    pub(crate) fn type_of_pat(
        &self,
        db: &'db dyn HirDatabase,
        pat: &ast::Pat,
    ) -> Option<(Type<'db>, Option<Type<'db>>)> {
        let expr_or_pat_id = self.pat_id(pat)?;
        let infer = self.infer()?;
        let coerced = match expr_or_pat_id {
            ExprOrPatId::ExprId(idx) => infer
                .expr_adjustment(idx)
                .and_then(|adjusts| adjusts.last().cloned())
                .map(|adjust| adjust.target),
            ExprOrPatId::PatId(idx) => {
                infer.pat_adjustment(idx).and_then(|adjusts| adjusts.last().cloned())
            }
        };

        let ty = infer[expr_or_pat_id].clone();
        let mk_ty = |ty| Type::new_with_resolver(db, &self.resolver, ty);
        Some((mk_ty(ty), coerced.map(mk_ty)))
    }

    pub(crate) fn type_of_binding_in_pat(
        &self,
        db: &'db dyn HirDatabase,
        pat: &ast::IdentPat,
    ) -> Option<Type<'db>> {
        let binding_id = self.binding_id_of_pat(pat)?;
        let infer = self.infer()?;
        let ty = infer[binding_id].clone();
        let mk_ty = |ty| Type::new_with_resolver(db, &self.resolver, ty);
        Some(mk_ty(ty))
    }

    pub(crate) fn type_of_self(
        &self,
        db: &'db dyn HirDatabase,
        _param: &ast::SelfParam,
    ) -> Option<Type<'db>> {
        let binding = self.body()?.self_param?;
        let ty = self.infer()?[binding].clone();
        Some(Type::new_with_resolver(db, &self.resolver, ty))
    }

    pub(crate) fn binding_mode_of_pat(
        &self,
        _db: &'db dyn HirDatabase,
        pat: &ast::IdentPat,
    ) -> Option<BindingMode> {
        let id = self.pat_id(&pat.clone().into())?;
        let infer = self.infer()?;
        infer.binding_mode(id.as_pat()?).map(|bm| match bm {
            hir_ty::BindingMode::Move => BindingMode::Move,
            hir_ty::BindingMode::Ref(hir_ty::Mutability::Mut) => BindingMode::Ref(Mutability::Mut),
            hir_ty::BindingMode::Ref(hir_ty::Mutability::Not) => {
                BindingMode::Ref(Mutability::Shared)
            }
        })
    }
    pub(crate) fn pattern_adjustments(
        &self,
        db: &'db dyn HirDatabase,
        pat: &ast::Pat,
    ) -> Option<SmallVec<[Type<'db>; 1]>> {
        let pat_id = self.pat_id(pat)?;
        let infer = self.infer()?;
        Some(
            infer
                .pat_adjustment(pat_id.as_pat()?)?
                .iter()
                .map(|ty| Type::new_with_resolver(db, &self.resolver, ty.clone()))
                .collect(),
        )
    }

    pub(crate) fn resolve_method_call_as_callable(
        &self,
        db: &'db dyn HirDatabase,
        call: &ast::MethodCallExpr,
    ) -> Option<Callable<'db>> {
        let expr_id = self.expr_id(call.clone().into())?.as_expr()?;
        let (func, substs) = self.infer()?.method_resolution(expr_id)?;
        let ty = db.value_ty(func.into())?.substitute(Interner, &substs);
        let ty = Type::new_with_resolver(db, &self.resolver, ty);
        let mut res = ty.as_callable(db)?;
        res.is_bound_method = true;
        Some(res)
    }

    pub(crate) fn resolve_method_call(
        &self,
        db: &'db dyn HirDatabase,
        call: &ast::MethodCallExpr,
    ) -> Option<Function> {
        let expr_id = self.expr_id(call.clone().into())?.as_expr()?;
        let (f_in_trait, substs) = self.infer()?.method_resolution(expr_id)?;

        Some(self.resolve_impl_method_or_trait_def(db, f_in_trait, substs).into())
    }

    pub(crate) fn resolve_method_call_fallback(
        &self,
        db: &'db dyn HirDatabase,
        call: &ast::MethodCallExpr,
    ) -> Option<(Either<Function, Field>, Option<GenericSubstitution<'db>>)> {
        let expr_id = self.expr_id(call.clone().into())?.as_expr()?;
        let inference_result = self.infer()?;
        match inference_result.method_resolution(expr_id) {
            Some((f_in_trait, substs)) => {
                let (fn_, subst) =
                    self.resolve_impl_method_or_trait_def_with_subst(db, f_in_trait, substs);
                Some((
                    Either::Left(fn_.into()),
                    Some(GenericSubstitution::new(fn_.into(), subst, self.trait_environment(db))),
                ))
            }
            None => {
                inference_result.field_resolution(expr_id).and_then(Either::left).map(|field| {
                    (Either::Right(field.into()), self.field_subst(expr_id, inference_result, db))
                })
            }
        }
    }

    pub(crate) fn resolve_expr_as_callable(
        &self,
        db: &'db dyn HirDatabase,
        call: &ast::Expr,
    ) -> Option<Callable<'db>> {
        let (orig, adjusted) = self.type_of_expr(db, &call.clone())?;
        adjusted.unwrap_or(orig).as_callable(db)
    }

    pub(crate) fn resolve_field(
        &self,
        field: &ast::FieldExpr,
    ) -> Option<Either<Field, TupleField>> {
        let (def, ..) = self.body_()?;
        let expr_id = self.expr_id(field.clone().into())?.as_expr()?;
        self.infer()?.field_resolution(expr_id).map(|it| {
            it.map_either(Into::into, |f| TupleField { owner: def, tuple: f.tuple, index: f.index })
        })
    }

    fn field_subst(
        &self,
        field_expr: ExprId,
        infer: &InferenceResult,
        db: &'db dyn HirDatabase,
    ) -> Option<GenericSubstitution<'db>> {
        let body = self.store()?;
        if let Expr::Field { expr: object_expr, name: _ } = body[field_expr] {
            let (adt, subst) = infer.type_of_expr_with_adjust(object_expr)?.as_adt()?;
            return Some(GenericSubstitution::new(
                adt.into(),
                subst.clone(),
                self.trait_environment(db),
            ));
        }
        None
    }

    pub(crate) fn resolve_field_fallback(
        &self,
        db: &'db dyn HirDatabase,
        field: &ast::FieldExpr,
    ) -> Option<(Either<Either<Field, TupleField>, Function>, Option<GenericSubstitution<'db>>)>
    {
        let (def, ..) = self.body_()?;
        let expr_id = self.expr_id(field.clone().into())?.as_expr()?;
        let inference_result = self.infer()?;
        match inference_result.field_resolution(expr_id) {
            Some(field) => match field {
                Either::Left(field) => Some((
                    Either::Left(Either::Left(field.into())),
                    self.field_subst(expr_id, inference_result, db),
                )),
                Either::Right(field) => Some((
                    Either::Left(Either::Right(TupleField {
                        owner: def,
                        tuple: field.tuple,
                        index: field.index,
                    })),
                    None,
                )),
            },
            None => inference_result.method_resolution(expr_id).map(|(f, substs)| {
                let (f, subst) = self.resolve_impl_method_or_trait_def_with_subst(db, f, substs);
                (
                    Either::Right(f.into()),
                    Some(GenericSubstitution::new(f.into(), subst, self.trait_environment(db))),
                )
            }),
        }
    }

    pub(crate) fn resolve_range_pat(
        &self,
        db: &'db dyn HirDatabase,
        range_pat: &ast::RangePat,
    ) -> Option<StructId> {
        let path: ModPath = match (range_pat.op_kind()?, range_pat.start(), range_pat.end()) {
            (RangeOp::Exclusive, None, Some(_)) => path![core::ops::RangeTo],
            (RangeOp::Exclusive, Some(_), None) => path![core::ops::RangeFrom],
            (RangeOp::Exclusive, Some(_), Some(_)) => path![core::ops::Range],
            (RangeOp::Inclusive, None, Some(_)) => path![core::ops::RangeToInclusive],
            (RangeOp::Inclusive, Some(_), Some(_)) => path![core::ops::RangeInclusive],

            (RangeOp::Exclusive, None, None) => return None,
            (RangeOp::Inclusive, None, None) => return None,
            (RangeOp::Inclusive, Some(_), None) => return None,
        };
        self.resolver.resolve_known_struct(db, &path)
    }

    pub(crate) fn resolve_range_expr(
        &self,
        db: &'db dyn HirDatabase,
        range_expr: &ast::RangeExpr,
    ) -> Option<StructId> {
        let path: ModPath = match (range_expr.op_kind()?, range_expr.start(), range_expr.end()) {
            (RangeOp::Exclusive, None, None) => path![core::ops::RangeFull],
            (RangeOp::Exclusive, None, Some(_)) => path![core::ops::RangeTo],
            (RangeOp::Exclusive, Some(_), None) => path![core::ops::RangeFrom],
            (RangeOp::Exclusive, Some(_), Some(_)) => path![core::ops::Range],
            (RangeOp::Inclusive, None, Some(_)) => path![core::ops::RangeToInclusive],
            (RangeOp::Inclusive, Some(_), Some(_)) => path![core::ops::RangeInclusive],

            // [E0586] inclusive ranges must be bounded at the end
            (RangeOp::Inclusive, None, None) => return None,
            (RangeOp::Inclusive, Some(_), None) => return None,
        };
        self.resolver.resolve_known_struct(db, &path)
    }

    pub(crate) fn resolve_await_to_poll(
        &self,
        db: &'db dyn HirDatabase,
        await_expr: &ast::AwaitExpr,
    ) -> Option<FunctionId> {
        let mut ty = self.ty_of_expr(await_expr.expr()?)?.clone();

        let into_future_trait = self
            .resolver
            .resolve_known_trait(db, &path![core::future::IntoFuture])
            .map(Trait::from);

        if let Some(into_future_trait) = into_future_trait {
            let type_ = Type::new_with_resolver(db, &self.resolver, ty.clone());
            if type_.impls_trait(db, into_future_trait, &[]) {
                let items = into_future_trait.items(db);
                let into_future_type = items.into_iter().find_map(|item| match item {
                    AssocItem::TypeAlias(alias)
                        if alias.name(db) == Name::new_symbol_root(sym::IntoFuture) =>
                    {
                        Some(alias)
                    }
                    _ => None,
                })?;
                let future_trait = type_.normalize_trait_assoc_type(db, &[], into_future_type)?;
                ty = future_trait.ty;
            }
        }

        let future_trait = LangItem::Future.resolve_trait(db, self.resolver.krate())?;
        let poll_fn = LangItem::FuturePoll.resolve_function(db, self.resolver.krate())?;
        // HACK: subst for `poll()` coincides with that for `Future` because `poll()` itself
        // doesn't have any generic parameters, so we skip building another subst for `poll()`.
        let substs = hir_ty::TyBuilder::subst_for_def(db, future_trait, None).push(ty).build();
        Some(self.resolve_impl_method_or_trait_def(db, poll_fn, substs))
    }

    pub(crate) fn resolve_prefix_expr(
        &self,
        db: &'db dyn HirDatabase,
        prefix_expr: &ast::PrefixExpr,
    ) -> Option<FunctionId> {
        let (op_trait, op_fn) = match prefix_expr.op_kind()? {
            ast::UnaryOp::Deref => {
                // This can be either `Deref::deref` or `DerefMut::deref_mut`.
                // Since deref kind is inferenced and stored in `InferenceResult.method_resolution`,
                // use that result to find out which one it is.
                let (deref_trait, deref) =
                    self.lang_trait_fn(db, LangItem::Deref, &Name::new_symbol_root(sym::deref))?;
                self.infer()
                    .and_then(|infer| {
                        let expr = self.expr_id(prefix_expr.clone().into())?.as_expr()?;
                        let (func, _) = infer.method_resolution(expr)?;
                        let (deref_mut_trait, deref_mut) = self.lang_trait_fn(
                            db,
                            LangItem::DerefMut,
                            &Name::new_symbol_root(sym::deref_mut),
                        )?;
                        if func == deref_mut { Some((deref_mut_trait, deref_mut)) } else { None }
                    })
                    .unwrap_or((deref_trait, deref))
            }
            ast::UnaryOp::Not => {
                self.lang_trait_fn(db, LangItem::Not, &Name::new_symbol_root(sym::not))?
            }
            ast::UnaryOp::Neg => {
                self.lang_trait_fn(db, LangItem::Neg, &Name::new_symbol_root(sym::neg))?
            }
        };

        let ty = self.ty_of_expr(prefix_expr.expr()?)?;

        // HACK: subst for all methods coincides with that for their trait because the methods
        // don't have any generic parameters, so we skip building another subst for the methods.
        let substs = hir_ty::TyBuilder::subst_for_def(db, op_trait, None).push(ty.clone()).build();

        Some(self.resolve_impl_method_or_trait_def(db, op_fn, substs))
    }

    pub(crate) fn resolve_index_expr(
        &self,
        db: &'db dyn HirDatabase,
        index_expr: &ast::IndexExpr,
    ) -> Option<FunctionId> {
        let base_ty = self.ty_of_expr(index_expr.base()?)?;
        let index_ty = self.ty_of_expr(index_expr.index()?)?;

        let (index_trait, index_fn) =
            self.lang_trait_fn(db, LangItem::Index, &Name::new_symbol_root(sym::index))?;
        let (op_trait, op_fn) = self
            .infer()
            .and_then(|infer| {
                let expr = self.expr_id(index_expr.clone().into())?.as_expr()?;
                let (func, _) = infer.method_resolution(expr)?;
                let (index_mut_trait, index_mut_fn) = self.lang_trait_fn(
                    db,
                    LangItem::IndexMut,
                    &Name::new_symbol_root(sym::index_mut),
                )?;
                if func == index_mut_fn { Some((index_mut_trait, index_mut_fn)) } else { None }
            })
            .unwrap_or((index_trait, index_fn));
        // HACK: subst for all methods coincides with that for their trait because the methods
        // don't have any generic parameters, so we skip building another subst for the methods.
        let substs = hir_ty::TyBuilder::subst_for_def(db, op_trait, None)
            .push(base_ty.clone())
            .push(index_ty.clone())
            .build();
        Some(self.resolve_impl_method_or_trait_def(db, op_fn, substs))
    }

    pub(crate) fn resolve_bin_expr(
        &self,
        db: &'db dyn HirDatabase,
        binop_expr: &ast::BinExpr,
    ) -> Option<FunctionId> {
        let op = binop_expr.op_kind()?;
        let lhs = self.ty_of_expr(binop_expr.lhs()?)?;
        let rhs = self.ty_of_expr(binop_expr.rhs()?)?;

        let (op_trait, op_fn) = lang_items_for_bin_op(op)
            .and_then(|(name, lang_item)| self.lang_trait_fn(db, lang_item, &name))?;
        // HACK: subst for `index()` coincides with that for `Index` because `index()` itself
        // doesn't have any generic parameters, so we skip building another subst for `index()`.
        let substs = hir_ty::TyBuilder::subst_for_def(db, op_trait, None)
            .push(lhs.clone())
            .push(rhs.clone())
            .build();

        Some(self.resolve_impl_method_or_trait_def(db, op_fn, substs))
    }

    pub(crate) fn resolve_try_expr(
        &self,
        db: &'db dyn HirDatabase,
        try_expr: &ast::TryExpr,
    ) -> Option<FunctionId> {
        let ty = self.ty_of_expr(try_expr.expr()?)?;

        let op_fn = LangItem::TryTraitBranch.resolve_function(db, self.resolver.krate())?;
        let op_trait = match op_fn.lookup(db).container {
            ItemContainerId::TraitId(id) => id,
            _ => return None,
        };
        // HACK: subst for `branch()` coincides with that for `Try` because `branch()` itself
        // doesn't have any generic parameters, so we skip building another subst for `branch()`.
        let substs = hir_ty::TyBuilder::subst_for_def(db, op_trait, None).push(ty.clone()).build();

        Some(self.resolve_impl_method_or_trait_def(db, op_fn, substs))
    }

    pub(crate) fn resolve_record_field(
        &self,
        db: &'db dyn HirDatabase,
        field: &ast::RecordExprField,
    ) -> Option<(Field, Option<Local>, Type<'db>, GenericSubstitution<'db>)> {
        let record_expr = ast::RecordExpr::cast(field.syntax().parent().and_then(|p| p.parent())?)?;
        let expr = ast::Expr::from(record_expr);
        let expr_id = self.store_sm()?.node_expr(InFile::new(self.file_id, &expr))?;

        let ast_name = field.field_name()?;
        let local_name = ast_name.as_name();
        let local = if field.name_ref().is_some() {
            None
        } else {
            // Shorthand syntax, resolve to the local
            let path = Path::from_known_path_with_no_generic(ModPath::from_segments(
                PathKind::Plain,
                once(local_name.clone()),
            ));
            match self.resolver.resolve_path_in_value_ns_fully(
                db,
                &path,
                name_hygiene(db, InFile::new(self.file_id, ast_name.syntax())),
            ) {
                Some(ValueNs::LocalBinding(binding_id)) => {
                    Some(Local { binding_id, parent: self.resolver.body_owner()? })
                }
                _ => None,
            }
        };
        let (adt, subst) = self.infer()?.type_of_expr_or_pat(expr_id)?.as_adt()?;
        let variant = self.infer()?.variant_resolution_for_expr_or_pat(expr_id)?;
        let variant_data = variant.fields(db);
        let field = FieldId { parent: variant, local_id: variant_data.field(&local_name)? };
        let field_ty =
            db.field_types(variant).get(field.local_id)?.clone().substitute(Interner, subst);
        Some((
            field.into(),
            local,
            Type::new_with_resolver(db, &self.resolver, field_ty),
            GenericSubstitution::new(adt.into(), subst.clone(), self.trait_environment(db)),
        ))
    }

    pub(crate) fn resolve_record_pat_field(
        &self,
        db: &'db dyn HirDatabase,
        field: &ast::RecordPatField,
    ) -> Option<(Field, Type<'db>, GenericSubstitution<'db>)> {
        let field_name = field.field_name()?.as_name();
        let record_pat = ast::RecordPat::cast(field.syntax().parent().and_then(|p| p.parent())?)?;
        let pat_id = self.pat_id(&record_pat.into())?;
        let variant = self.infer()?.variant_resolution_for_pat(pat_id.as_pat()?)?;
        let variant_data = variant.fields(db);
        let field = FieldId { parent: variant, local_id: variant_data.field(&field_name)? };
        let (adt, subst) = self.infer()?[pat_id.as_pat()?].as_adt()?;
        let field_ty =
            db.field_types(variant).get(field.local_id)?.clone().substitute(Interner, subst);
        Some((
            field.into(),
            Type::new_with_resolver(db, &self.resolver, field_ty),
            GenericSubstitution::new(adt.into(), subst.clone(), self.trait_environment(db)),
        ))
    }

    pub(crate) fn resolve_bind_pat_to_const(
        &self,
        db: &'db dyn HirDatabase,
        pat: &ast::IdentPat,
    ) -> Option<ModuleDef> {
        let expr_or_pat_id = self.pat_id(&pat.clone().into())?;
        let store = self.store()?;

        let path = match expr_or_pat_id {
            ExprOrPatId::ExprId(idx) => match &store[idx] {
                Expr::Path(path) => path,
                _ => return None,
            },
            ExprOrPatId::PatId(idx) => match &store[idx] {
                Pat::Path(path) => path,
                _ => return None,
            },
        };

        let body_owner = self.resolver.body_owner();
        let res = resolve_hir_value_path(db, &self.resolver, body_owner, path, HygieneId::ROOT)?;
        match res {
            PathResolution::Def(def) => Some(def),
            _ => None,
        }
    }

    pub(crate) fn resolve_use_type_arg(&self, name: &ast::NameRef) -> Option<crate::TypeParam> {
        let name = name.as_name();
        self.resolver
            .all_generic_params()
            .find_map(|(params, parent)| params.find_type_by_name(&name, *parent))
            .map(crate::TypeParam::from)
    }

    pub(crate) fn resolve_offset_of_field(
        &self,
        db: &'db dyn HirDatabase,
        name_ref: &ast::NameRef,
    ) -> Option<(Either<crate::Variant, crate::Field>, GenericSubstitution<'db>)> {
        let offset_of_expr = ast::OffsetOfExpr::cast(name_ref.syntax().parent()?)?;
        let container = offset_of_expr.ty()?;
        let container = self.type_of_type(db, &container)?;

        let trait_env = container.env;
        let mut container = Either::Right(container.ty);
        for field_name in offset_of_expr.fields() {
            if let Some(
                TyKind::Alias(AliasTy::Projection(ProjectionTy { associated_ty_id, substitution }))
                | TyKind::AssociatedType(associated_ty_id, substitution),
            ) = container.as_ref().right().map(|it| it.kind(Interner))
            {
                let projection = ProjectionTy {
                    associated_ty_id: *associated_ty_id,
                    substitution: substitution.clone(),
                };
                container = Either::Right(db.normalize_projection(projection, trait_env.clone()));
            }
            let handle_variants = |variant: VariantId, subst: &Substitution, container: &mut _| {
                let fields = variant.fields(db);
                let field = fields.field(&field_name.as_name())?;
                let field_types = db.field_types(variant);
                *container = Either::Right(field_types[field].clone().substitute(Interner, subst));
                let generic_def = match variant {
                    VariantId::EnumVariantId(it) => it.loc(db).parent.into(),
                    VariantId::StructId(it) => it.into(),
                    VariantId::UnionId(it) => it.into(),
                };
                Some((
                    Either::Right(Field { parent: variant.into(), id: field }),
                    generic_def,
                    subst.clone(),
                ))
            };
            let temp_ty = TyKind::Error.intern(Interner);
            let (field_def, generic_def, subst) =
                match std::mem::replace(&mut container, Either::Right(temp_ty.clone())) {
                    Either::Left((variant_id, subst)) => {
                        handle_variants(VariantId::from(variant_id), &subst, &mut container)?
                    }
                    Either::Right(container_ty) => match container_ty.kind(Interner) {
                        TyKind::Adt(adt_id, subst) => match adt_id.0 {
                            AdtId::StructId(id) => {
                                handle_variants(id.into(), subst, &mut container)?
                            }
                            AdtId::UnionId(id) => {
                                handle_variants(id.into(), subst, &mut container)?
                            }
                            AdtId::EnumId(id) => {
                                let variants = id.enum_variants(db);
                                let variant = variants.variant(&field_name.as_name())?;
                                container = Either::Left((variant, subst.clone()));
                                (Either::Left(Variant { id: variant }), id.into(), subst.clone())
                            }
                        },
                        _ => return None,
                    },
                };

            if field_name.syntax().text_range() == name_ref.syntax().text_range() {
                return Some((field_def, GenericSubstitution::new(generic_def, subst, trait_env)));
            }
        }
        never!("the `NameRef` is a child of the `OffsetOfExpr`, we should've visited it");
        None
    }

    pub(crate) fn resolve_path(
        &self,
        db: &'db dyn HirDatabase,
        path: &ast::Path,
    ) -> Option<(PathResolution, Option<GenericSubstitution<'db>>)> {
        let parent = path.syntax().parent();
        let parent = || parent.clone();

        let mut prefer_value_ns = false;
        let resolved = (|| {
            let infer = self.infer()?;
            if let Some(path_expr) = parent().and_then(ast::PathExpr::cast) {
                let expr_id = self.expr_id(path_expr.into())?;
                if let Some((assoc, subs)) = infer.assoc_resolutions_for_expr_or_pat(expr_id) {
                    let (assoc, subst) = match assoc {
                        AssocItemId::FunctionId(f_in_trait) => {
                            match infer.type_of_expr_or_pat(expr_id) {
                                None => {
                                    let subst = GenericSubstitution::new(
                                        f_in_trait.into(),
                                        subs,
                                        self.trait_environment(db),
                                    );
                                    (assoc, subst)
                                }
                                Some(func_ty) => {
                                    if let TyKind::FnDef(_fn_def, subs) = func_ty.kind(Interner) {
                                        let (fn_, subst) = self
                                            .resolve_impl_method_or_trait_def_with_subst(
                                                db,
                                                f_in_trait,
                                                subs.clone(),
                                            );
                                        let subst = GenericSubstitution::new(
                                            fn_.into(),
                                            subst,
                                            self.trait_environment(db),
                                        );
                                        (fn_.into(), subst)
                                    } else {
                                        let subst = GenericSubstitution::new(
                                            f_in_trait.into(),
                                            subs,
                                            self.trait_environment(db),
                                        );
                                        (assoc, subst)
                                    }
                                }
                            }
                        }
                        AssocItemId::ConstId(const_id) => {
                            let (konst, subst) =
                                self.resolve_impl_const_or_trait_def_with_subst(db, const_id, subs);
                            let subst = GenericSubstitution::new(
                                konst.into(),
                                subst,
                                self.trait_environment(db),
                            );
                            (konst.into(), subst)
                        }
                        AssocItemId::TypeAliasId(type_alias) => (
                            assoc,
                            GenericSubstitution::new(
                                type_alias.into(),
                                subs,
                                self.trait_environment(db),
                            ),
                        ),
                    };

                    return Some((PathResolution::Def(AssocItem::from(assoc).into()), Some(subst)));
                }
                if let Some(VariantId::EnumVariantId(variant)) =
                    infer.variant_resolution_for_expr_or_pat(expr_id)
                {
                    return Some((PathResolution::Def(ModuleDef::Variant(variant.into())), None));
                }
                prefer_value_ns = true;
            } else if let Some(path_pat) = parent().and_then(ast::PathPat::cast) {
                let expr_or_pat_id = self.pat_id(&path_pat.into())?;
                if let Some((assoc, subs)) = infer.assoc_resolutions_for_expr_or_pat(expr_or_pat_id)
                {
                    let (assoc, subst) = match assoc {
                        AssocItemId::ConstId(const_id) => {
                            let (konst, subst) =
                                self.resolve_impl_const_or_trait_def_with_subst(db, const_id, subs);
                            let subst = GenericSubstitution::new(
                                konst.into(),
                                subst,
                                self.trait_environment(db),
                            );
                            (konst.into(), subst)
                        }
                        assoc => (
                            assoc,
                            GenericSubstitution::new(
                                assoc.into(),
                                subs,
                                self.trait_environment(db),
                            ),
                        ),
                    };
                    return Some((PathResolution::Def(AssocItem::from(assoc).into()), Some(subst)));
                }
                if let Some(VariantId::EnumVariantId(variant)) =
                    infer.variant_resolution_for_expr_or_pat(expr_or_pat_id)
                {
                    return Some((PathResolution::Def(ModuleDef::Variant(variant.into())), None));
                }
            } else if let Some(rec_lit) = parent().and_then(ast::RecordExpr::cast) {
                let expr_id = self.expr_id(rec_lit.into())?;
                if let Some(VariantId::EnumVariantId(variant)) =
                    infer.variant_resolution_for_expr_or_pat(expr_id)
                {
                    return Some((PathResolution::Def(ModuleDef::Variant(variant.into())), None));
                }
            } else {
                let record_pat = parent().and_then(ast::RecordPat::cast).map(ast::Pat::from);
                let tuple_struct_pat =
                    || parent().and_then(ast::TupleStructPat::cast).map(ast::Pat::from);
                if let Some(pat) = record_pat.or_else(tuple_struct_pat) {
                    let pat_id = self.pat_id(&pat)?;
                    let variant_res_for_pat = infer.variant_resolution_for_pat(pat_id.as_pat()?);
                    if let Some(VariantId::EnumVariantId(variant)) = variant_res_for_pat {
                        return Some((
                            PathResolution::Def(ModuleDef::Variant(variant.into())),
                            None,
                        ));
                    }
                }
            }
            None
        })();
        if resolved.is_some() {
            return resolved;
        }

        // FIXME: collectiong here shouldnt be necessary?
        let mut collector = ExprCollector::new(db, self.resolver.module(), self.file_id);
        let hir_path =
            collector.lower_path(path.clone(), &mut ExprCollector::impl_trait_error_allocator)?;
        let parent_hir_path = path
            .parent_path()
            .and_then(|p| collector.lower_path(p, &mut ExprCollector::impl_trait_error_allocator));
        let (store, _) = collector.store.finish();

        // Case where path is a qualifier of a use tree, e.g. foo::bar::{Baz, Qux} where we are
        // trying to resolve foo::bar.
        if let Some(use_tree) = parent().and_then(ast::UseTree::cast)
            && use_tree.coloncolon_token().is_some()
        {
            return resolve_hir_path_qualifier(db, &self.resolver, &hir_path, &store)
                .map(|it| (it, None));
        }

        let meta_path = path
            .syntax()
            .ancestors()
            .take_while(|it| {
                let kind = it.kind();
                ast::Path::can_cast(kind) || ast::Meta::can_cast(kind)
            })
            .last()
            .and_then(ast::Meta::cast);

        // Case where path is a qualifier of another path, e.g. foo::bar::Baz where we are
        // trying to resolve foo::bar.
        if let Some(parent_hir_path) = parent_hir_path {
            return match resolve_hir_path_qualifier(db, &self.resolver, &hir_path, &store) {
                None if meta_path.is_some() => path
                    .first_segment()
                    .and_then(|it| it.name_ref())
                    .and_then(|name_ref| {
                        ToolModule::by_name(db, self.resolver.krate().into(), &name_ref.text())
                            .map(PathResolution::ToolModule)
                    })
                    .map(|it| (it, None)),
                // Case the type name conflict with use module,
                // e.g.
                // ```
                // use std::str;
                // fn main() {
                //     str::from_utf8();  // as module std::str
                //     str::len();        // as primitive type str
                //     str::no_exist_item(); // as primitive type str
                // }
                // ```
                Some(it) if matches!(it, PathResolution::Def(ModuleDef::BuiltinType(_))) => {
                    if let Some(mod_path) = hir_path.mod_path()
                        && let Some(ModuleDefId::ModuleId(id)) =
                            self.resolver.resolve_module_path_in_items(db, mod_path).take_types()
                    {
                        let parent_hir_name = parent_hir_path.segments().get(1).map(|it| it.name);
                        let module = crate::Module { id };
                        if module
                            .scope(db, None)
                            .into_iter()
                            .any(|(name, _)| Some(&name) == parent_hir_name)
                        {
                            return Some((PathResolution::Def(ModuleDef::Module(module)), None));
                        };
                    }
                    Some((it, None))
                }
                // FIXME: We do not show substitutions for parts of path, because this is really complex
                // due to the interactions with associated items of `impl`s and associated items of associated
                // types.
                res => res.map(|it| (it, None)),
            };
        } else if let Some(meta_path) = meta_path {
            // Case where we are resolving the final path segment of a path in an attribute
            // in this case we have to check for inert/builtin attributes and tools and prioritize
            // resolution of attributes over other namespaces
            if let Some(name_ref) = path.as_single_name_ref() {
                let builtin =
                    BuiltinAttr::by_name(db, self.resolver.krate().into(), &name_ref.text());
                if builtin.is_some() {
                    return builtin.map(|it| (PathResolution::BuiltinAttr(it), None));
                }

                if let Some(attr) = meta_path.parent_attr() {
                    let adt = if let Some(field) =
                        attr.syntax().parent().and_then(ast::RecordField::cast)
                    {
                        field.syntax().ancestors().take(4).find_map(ast::Adt::cast)
                    } else if let Some(field) =
                        attr.syntax().parent().and_then(ast::TupleField::cast)
                    {
                        field.syntax().ancestors().take(4).find_map(ast::Adt::cast)
                    } else if let Some(variant) =
                        attr.syntax().parent().and_then(ast::Variant::cast)
                    {
                        variant.syntax().ancestors().nth(2).and_then(ast::Adt::cast)
                    } else {
                        None
                    };
                    if let Some(adt) = adt {
                        let ast_id = db.ast_id_map(self.file_id).ast_id(&adt);
                        if let Some(helpers) = self
                            .resolver
                            .def_map()
                            .derive_helpers_in_scope(InFile::new(self.file_id, ast_id))
                        {
                            // FIXME: Multiple derives can have the same helper
                            let name_ref = name_ref.as_name();
                            for (macro_id, mut helpers) in
                                helpers.iter().chunk_by(|(_, macro_id, ..)| macro_id).into_iter()
                            {
                                if let Some(idx) = helpers.position(|(name, ..)| *name == name_ref)
                                {
                                    return Some((
                                        PathResolution::DeriveHelper(DeriveHelper {
                                            derive: *macro_id,
                                            idx: idx as u32,
                                        }),
                                        None,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            return match resolve_hir_path_as_attr_macro(db, &self.resolver, &hir_path) {
                Some(m) => Some((PathResolution::Def(ModuleDef::Macro(m)), None)),
                // this labels any path that starts with a tool module as the tool itself, this is technically wrong
                // but there is no benefit in differentiating these two cases for the time being
                None => path
                    .first_segment()
                    .and_then(|it| it.name_ref())
                    .and_then(|name_ref| {
                        ToolModule::by_name(db, self.resolver.krate().into(), &name_ref.text())
                            .map(PathResolution::ToolModule)
                    })
                    .map(|it| (it, None)),
            };
        }
        if parent().is_some_and(|it| ast::Visibility::can_cast(it.kind())) {
            // No substitution because only modules can be inside visibilities, and those have no generics.
            resolve_hir_path_qualifier(db, &self.resolver, &hir_path, &store).map(|it| (it, None))
        } else {
            // Probably a type, no need to show substitutions for those.
            let res = resolve_hir_path_(
                db,
                &self.resolver,
                &hir_path,
                prefer_value_ns,
                name_hygiene(db, InFile::new(self.file_id, path.syntax())),
                Some(&store),
                false,
            )
            .any()?;
            let subst = (|| {
                let parent = parent()?;
                let ty = if let Some(expr) = ast::Expr::cast(parent.clone()) {
                    let expr_id = self.expr_id(expr)?;
                    self.infer()?.type_of_expr_or_pat(expr_id)?
                } else if let Some(pat) = ast::Pat::cast(parent) {
                    let pat_id = self.pat_id(&pat)?;
                    &self.infer()?[pat_id]
                } else {
                    return None;
                };
                let env = self.trait_environment(db);
                let (subst, expected_resolution) = match ty.kind(Interner) {
                    TyKind::Adt(adt_id, subst) => (
                        GenericSubstitution::new(adt_id.0.into(), subst.clone(), env),
                        PathResolution::Def(ModuleDef::Adt(adt_id.0.into())),
                    ),
                    TyKind::AssociatedType(assoc_id, subst) => {
                        let assoc_id = from_assoc_type_id(*assoc_id);
                        (
                            GenericSubstitution::new(assoc_id.into(), subst.clone(), env),
                            PathResolution::Def(ModuleDef::TypeAlias(assoc_id.into())),
                        )
                    }
                    TyKind::FnDef(fn_id, subst) => {
                        let fn_id = ToChalk::from_chalk(db, *fn_id);
                        let generic_def_id = match fn_id {
                            CallableDefId::StructId(id) => id.into(),
                            CallableDefId::FunctionId(id) => id.into(),
                            CallableDefId::EnumVariantId(_) => return None,
                        };
                        (
                            GenericSubstitution::new(generic_def_id, subst.clone(), env),
                            PathResolution::Def(ModuleDefId::from(fn_id).into()),
                        )
                    }
                    _ => return None,
                };
                if res != expected_resolution {
                    // The user will not understand where we're coming from. This can happen (I think) with type aliases.
                    return None;
                }
                Some(subst)
            })();
            Some((res, subst))
        }
    }

    pub(crate) fn resolve_hir_path_per_ns(
        &self,
        db: &dyn HirDatabase,
        path: &ast::Path,
    ) -> Option<PathResolutionPerNs> {
        let mut collector = ExprCollector::new(db, self.resolver.module(), self.file_id);
        let hir_path =
            collector.lower_path(path.clone(), &mut ExprCollector::impl_trait_error_allocator)?;
        let (store, _) = collector.store.finish();
        Some(resolve_hir_path_(
            db,
            &self.resolver,
            &hir_path,
            false,
            name_hygiene(db, InFile::new(self.file_id, path.syntax())),
            Some(&store),
            true,
        ))
    }

    pub(crate) fn record_literal_missing_fields(
        &self,
        db: &'db dyn HirDatabase,
        literal: &ast::RecordExpr,
    ) -> Option<Vec<(Field, Type<'db>)>> {
        let body = self.store()?;
        let infer = self.infer()?;

        let expr_id = self.expr_id(literal.clone().into())?;
        let substs = infer[expr_id].as_adt()?.1;

        let (variant, missing_fields, _exhaustive) = match expr_id {
            ExprOrPatId::ExprId(expr_id) => {
                record_literal_missing_fields(db, infer, expr_id, &body[expr_id])?
            }
            ExprOrPatId::PatId(pat_id) => {
                record_pattern_missing_fields(db, infer, pat_id, &body[pat_id])?
            }
        };
        let res = self.missing_fields(db, substs, variant, missing_fields);
        Some(res)
    }

    pub(crate) fn record_pattern_missing_fields(
        &self,
        db: &'db dyn HirDatabase,
        pattern: &ast::RecordPat,
    ) -> Option<Vec<(Field, Type<'db>)>> {
        let body = self.store()?;
        let infer = self.infer()?;

        let pat_id = self.pat_id(&pattern.clone().into())?.as_pat()?;
        let substs = infer[pat_id].as_adt()?.1;

        let (variant, missing_fields, _exhaustive) =
            record_pattern_missing_fields(db, infer, pat_id, &body[pat_id])?;
        let res = self.missing_fields(db, substs, variant, missing_fields);
        Some(res)
    }

    fn missing_fields(
        &self,
        db: &'db dyn HirDatabase,
        substs: &Substitution,
        variant: VariantId,
        missing_fields: Vec<LocalFieldId>,
    ) -> Vec<(Field, Type<'db>)> {
        let field_types = db.field_types(variant);

        missing_fields
            .into_iter()
            .map(|local_id| {
                let field = FieldId { parent: variant, local_id };
                let ty = field_types[local_id].clone().substitute(Interner, substs);
                (field.into(), Type::new_with_resolver_inner(db, &self.resolver, ty))
            })
            .collect()
    }

    pub(crate) fn resolve_variant(&self, record_lit: ast::RecordExpr) -> Option<VariantId> {
        let infer = self.infer()?;
        let expr_id = self.expr_id(record_lit.into())?;
        infer.variant_resolution_for_expr_or_pat(expr_id)
    }

    pub(crate) fn is_unsafe_macro_call_expr(
        &self,
        db: &'db dyn HirDatabase,
        macro_expr: InFile<&ast::MacroExpr>,
    ) -> bool {
        if let Some((def, body, sm, Some(infer))) = self.body_()
            && let Some(expanded_expr) = sm.macro_expansion_expr(macro_expr)
        {
            let mut is_unsafe = false;
            let mut walk_expr = |expr_id| {
                unsafe_operations(db, infer, def, body, expr_id, &mut |inside_unsafe_block| {
                    is_unsafe |= inside_unsafe_block == InsideUnsafeBlock::No
                })
            };
            match expanded_expr {
                ExprOrPatId::ExprId(expanded_expr) => walk_expr(expanded_expr),
                ExprOrPatId::PatId(expanded_pat) => {
                    body.walk_exprs_in_pat(expanded_pat, &mut walk_expr)
                }
            }
            return is_unsafe;
        }
        false
    }

    /// Returns the range of the implicit template argument and its resolution at the given `offset`
    pub(crate) fn resolve_offset_in_format_args(
        &self,
        db: &'db dyn HirDatabase,
        format_args: InFile<&ast::FormatArgsExpr>,
        offset: TextSize,
    ) -> Option<(TextRange, Option<PathResolution>)> {
        let (hygiene, implicits) = self.store_sm()?.implicit_format_args(format_args)?;
        implicits.iter().find(|(range, _)| range.contains_inclusive(offset)).map(|(range, name)| {
            (
                *range,
                resolve_hir_value_path(
                    db,
                    &self.resolver,
                    self.resolver.body_owner(),
                    &Path::from_known_path_with_no_generic(ModPath::from_segments(
                        PathKind::Plain,
                        Some(name.clone()),
                    )),
                    hygiene,
                ),
            )
        })
    }

    pub(crate) fn resolve_offset_in_asm_template(
        &self,
        asm: InFile<&ast::AsmExpr>,
        line: usize,
        offset: TextSize,
    ) -> Option<(DefWithBodyId, (ExprId, TextRange, usize))> {
        let (def, _, body_source_map, _) = self.body_()?;
        let (expr, args) = body_source_map.asm_template_args(asm)?;
        Some(def).zip(
            args.get(line)?
                .iter()
                .find(|(range, _)| range.contains_inclusive(offset))
                .map(|(range, idx)| (expr, *range, *idx)),
        )
    }

    pub(crate) fn as_format_args_parts<'a>(
        &'a self,
        db: &'a dyn HirDatabase,
        format_args: InFile<&ast::FormatArgsExpr>,
    ) -> Option<impl Iterator<Item = (TextRange, Option<PathResolution>)> + 'a> {
        let (hygiene, names) = self.store_sm()?.implicit_format_args(format_args)?;
        Some(names.iter().map(move |(range, name)| {
            (
                *range,
                resolve_hir_value_path(
                    db,
                    &self.resolver,
                    self.resolver.body_owner(),
                    &Path::from_known_path_with_no_generic(ModPath::from_segments(
                        PathKind::Plain,
                        Some(name.clone()),
                    )),
                    hygiene,
                ),
            )
        }))
    }

    pub(crate) fn as_asm_parts(
        &self,
        asm: InFile<&ast::AsmExpr>,
    ) -> Option<(DefWithBodyId, (ExprId, &[Vec<(TextRange, usize)>]))> {
        let (def, _, body_source_map, _) = self.body_()?;
        Some(def).zip(body_source_map.asm_template_args(asm))
    }

    fn resolve_impl_method_or_trait_def(
        &self,
        db: &'db dyn HirDatabase,
        func: FunctionId,
        substs: Substitution,
    ) -> FunctionId {
        self.resolve_impl_method_or_trait_def_with_subst(db, func, substs).0
    }

    fn resolve_impl_method_or_trait_def_with_subst(
        &self,
        db: &'db dyn HirDatabase,
        func: FunctionId,
        substs: Substitution,
    ) -> (FunctionId, Substitution) {
        let owner = match self.resolver.body_owner() {
            Some(it) => it,
            None => return (func, substs),
        };
        let env = db.trait_environment_for_body(owner);
        db.lookup_impl_method(env, func, substs)
    }

    fn resolve_impl_const_or_trait_def_with_subst(
        &self,
        db: &'db dyn HirDatabase,
        const_id: ConstId,
        subs: Substitution,
    ) -> (ConstId, Substitution) {
        let owner = match self.resolver.body_owner() {
            Some(it) => it,
            None => return (const_id, subs),
        };
        let env = db.trait_environment_for_body(owner);
        method_resolution::lookup_impl_const(db, env, const_id, subs)
    }

    fn lang_trait_fn(
        &self,
        db: &'db dyn HirDatabase,
        lang_trait: LangItem,
        method_name: &Name,
    ) -> Option<(TraitId, FunctionId)> {
        let trait_id = lang_trait.resolve_trait(db, self.resolver.krate())?;
        let fn_id = trait_id.trait_items(db).method_by_name(method_name)?;
        Some((trait_id, fn_id))
    }

    fn ty_of_expr(&self, expr: ast::Expr) -> Option<&Ty> {
        self.infer()?.type_of_expr_or_pat(self.expr_id(expr)?)
    }
}

fn scope_for(
    db: &dyn HirDatabase,
    scopes: &ExprScopes,
    source_map: &BodySourceMap,
    node: InFile<&SyntaxNode>,
) -> Option<ScopeId> {
    node.ancestors_with_macros(db)
        .take_while(|it| {
            let kind = it.kind();
            !ast::Item::can_cast(kind)
                || ast::MacroCall::can_cast(kind)
                || ast::Use::can_cast(kind)
                || ast::AsmExpr::can_cast(kind)
        })
        .filter_map(|it| it.map(ast::Expr::cast).transpose())
        .filter_map(|it| source_map.node_expr(it.as_ref())?.as_expr())
        .find_map(|it| scopes.scope_for(it))
}

fn scope_for_offset(
    db: &dyn HirDatabase,
    scopes: &ExprScopes,
    source_map: &BodySourceMap,
    from_file: HirFileId,
    offset: TextSize,
) -> Option<ScopeId> {
    scopes
        .scope_by_expr()
        .iter()
        .filter_map(|(id, scope)| {
            let InFile { file_id, value } = source_map.expr_syntax(id).ok()?;
            if from_file == file_id {
                return Some((value.text_range(), scope));
            }

            // FIXME handle attribute expansion
            let source = iter::successors(file_id.macro_file().map(|it| it.call_node(db)), |it| {
                Some(it.file_id.macro_file()?.call_node(db))
            })
            .find(|it| it.file_id == from_file)
            .filter(|it| it.kind() == SyntaxKind::MACRO_CALL)?;
            Some((source.text_range(), scope))
        })
        .filter(|(expr_range, _scope)| expr_range.start() <= offset && offset <= expr_range.end())
        // find containing scope
        .min_by_key(|(expr_range, _scope)| expr_range.len())
        .map(|(expr_range, scope)| {
            adjust(db, scopes, source_map, expr_range, from_file, offset).unwrap_or(*scope)
        })
}

// XXX: during completion, cursor might be outside of any particular
// expression. Try to figure out the correct scope...
fn adjust(
    db: &dyn HirDatabase,
    scopes: &ExprScopes,
    source_map: &BodySourceMap,
    expr_range: TextRange,
    from_file: HirFileId,
    offset: TextSize,
) -> Option<ScopeId> {
    let child_scopes = scopes
        .scope_by_expr()
        .iter()
        .filter_map(|(id, scope)| {
            let source = source_map.expr_syntax(id).ok()?;
            // FIXME: correctly handle macro expansion
            if source.file_id != from_file {
                return None;
            }
            let root = source.file_syntax(db);
            let node = source.value.to_node(&root);
            Some((node.syntax().text_range(), scope))
        })
        .filter(|&(range, _)| {
            range.start() <= offset && expr_range.contains_range(range) && range != expr_range
        });

    child_scopes
        .max_by(|&(r1, _), &(r2, _)| {
            if r1.contains_range(r2) {
                std::cmp::Ordering::Greater
            } else if r2.contains_range(r1) {
                std::cmp::Ordering::Less
            } else {
                r1.start().cmp(&r2.start())
            }
        })
        .map(|(_ptr, scope)| *scope)
}

#[inline]
pub(crate) fn resolve_hir_path(
    db: &dyn HirDatabase,
    resolver: &Resolver<'_>,
    path: &Path,
    hygiene: HygieneId,
    store: Option<&ExpressionStore>,
) -> Option<PathResolution> {
    resolve_hir_path_(db, resolver, path, false, hygiene, store, false).any()
}

#[inline]
pub(crate) fn resolve_hir_path_as_attr_macro(
    db: &dyn HirDatabase,
    resolver: &Resolver<'_>,
    path: &Path,
) -> Option<Macro> {
    resolver
        .resolve_path_as_macro(db, path.mod_path()?, Some(MacroSubNs::Attr))
        .map(|(it, _)| it)
        .map(Into::into)
}

fn resolve_hir_path_(
    db: &dyn HirDatabase,
    resolver: &Resolver<'_>,
    path: &Path,
    prefer_value_ns: bool,
    hygiene: HygieneId,
    store: Option<&ExpressionStore>,
    resolve_per_ns: bool,
) -> PathResolutionPerNs {
    let types = || {
        let (ty, unresolved) = match path.type_anchor() {
            Some(type_ref) => resolver.generic_def().and_then(|def| {
                let (_, res) =
                    TyLoweringContext::new(db, resolver, store?, def, LifetimeElisionKind::Infer)
                        .lower_ty_ext(type_ref);
                res.map(|ty_ns| (ty_ns, path.segments().first()))
            }),
            None => {
                let (ty, remaining_idx, _) = resolver.resolve_path_in_type_ns(db, path)?;
                match remaining_idx {
                    Some(remaining_idx) => {
                        if remaining_idx + 1 == path.segments().len() {
                            Some((ty, path.segments().last()))
                        } else {
                            None
                        }
                    }
                    None => Some((ty, None)),
                }
            }
        }?;

        // If we are in a TypeNs for a Trait, and we have an unresolved name, try to resolve it as a type
        // within the trait's associated types.
        if let (Some(unresolved), &TypeNs::TraitId(trait_id)) = (&unresolved, &ty)
            && let Some(type_alias_id) =
                trait_id.trait_items(db).associated_type_by_name(unresolved.name)
        {
            return Some(PathResolution::Def(ModuleDefId::from(type_alias_id).into()));
        }

        let res = match ty {
            TypeNs::SelfType(it) => PathResolution::SelfType(it.into()),
            TypeNs::GenericParam(id) => PathResolution::TypeParam(id.into()),
            TypeNs::AdtSelfType(it) | TypeNs::AdtId(it) => {
                PathResolution::Def(Adt::from(it).into())
            }
            TypeNs::EnumVariantId(it) => PathResolution::Def(Variant::from(it).into()),
            TypeNs::TypeAliasId(it) => PathResolution::Def(TypeAlias::from(it).into()),
            TypeNs::BuiltinType(it) => PathResolution::Def(BuiltinType::from(it).into()),
            TypeNs::TraitId(it) => PathResolution::Def(Trait::from(it).into()),
            TypeNs::TraitAliasId(it) => PathResolution::Def(TraitAlias::from(it).into()),
            TypeNs::ModuleId(it) => PathResolution::Def(ModuleDef::Module(it.into())),
        };
        match unresolved {
            Some(unresolved) => resolver
                .generic_def()
                .and_then(|def| {
                    hir_ty::associated_type_shorthand_candidates(
                        db,
                        def,
                        res.in_type_ns()?,
                        |name, id| (name == unresolved.name).then_some(id),
                    )
                })
                .map(TypeAlias::from)
                .map(Into::into)
                .map(PathResolution::Def),
            None => Some(res),
        }
    };

    let body_owner = resolver.body_owner();
    let values = || resolve_hir_value_path(db, resolver, body_owner, path, hygiene);

    let items = || {
        resolver
            .resolve_module_path_in_items(db, path.mod_path()?)
            .take_types()
            .map(|it| PathResolution::Def(it.into()))
    };

    let macros = || {
        resolver
            .resolve_path_as_macro(db, path.mod_path()?, None)
            .map(|(def, _)| PathResolution::Def(ModuleDef::Macro(def.into())))
    };

    if resolve_per_ns {
        PathResolutionPerNs {
            type_ns: types().or_else(items),
            value_ns: values(),
            macro_ns: macros(),
        }
    } else {
        let res = if prefer_value_ns {
            values()
                .map(|value_ns| PathResolutionPerNs::new(None, Some(value_ns), None))
                .unwrap_or_else(|| PathResolutionPerNs::new(types(), None, None))
        } else {
            types()
                .map(|type_ns| PathResolutionPerNs::new(Some(type_ns), None, None))
                .unwrap_or_else(|| PathResolutionPerNs::new(None, values(), None))
        };

        if res.any().is_some() {
            res
        } else if let Some(type_ns) = items() {
            PathResolutionPerNs::new(Some(type_ns), None, None)
        } else {
            PathResolutionPerNs::new(None, None, macros())
        }
    }
}

fn resolve_hir_value_path(
    db: &dyn HirDatabase,
    resolver: &Resolver<'_>,
    body_owner: Option<DefWithBodyId>,
    path: &Path,
    hygiene: HygieneId,
) -> Option<PathResolution> {
    resolver.resolve_path_in_value_ns_fully(db, path, hygiene).and_then(|val| {
        let res = match val {
            ValueNs::LocalBinding(binding_id) => {
                let var = Local { parent: body_owner?, binding_id };
                PathResolution::Local(var)
            }
            ValueNs::FunctionId(it) => PathResolution::Def(Function::from(it).into()),
            ValueNs::ConstId(it) => PathResolution::Def(Const::from(it).into()),
            ValueNs::StaticId(it) => PathResolution::Def(Static::from(it).into()),
            ValueNs::StructId(it) => PathResolution::Def(Struct::from(it).into()),
            ValueNs::EnumVariantId(it) => PathResolution::Def(Variant::from(it).into()),
            ValueNs::ImplSelf(impl_id) => PathResolution::SelfType(impl_id.into()),
            ValueNs::GenericParam(id) => PathResolution::ConstParam(id.into()),
        };
        Some(res)
    })
}

/// Resolves a path where we know it is a qualifier of another path.
///
/// For example, if we have:
/// ```
/// mod my {
///     pub mod foo {
///         struct Bar;
///     }
///
///     pub fn foo() {}
/// }
/// ```
/// then we know that `foo` in `my::foo::Bar` refers to the module, not the function.
fn resolve_hir_path_qualifier(
    db: &dyn HirDatabase,
    resolver: &Resolver<'_>,
    path: &Path,
    store: &ExpressionStore,
) -> Option<PathResolution> {
    (|| {
        let (ty, unresolved) = match path.type_anchor() {
            Some(type_ref) => resolver.generic_def().and_then(|def| {
                let (_, res) =
                    TyLoweringContext::new(db, resolver, store, def, LifetimeElisionKind::Infer)
                        .lower_ty_ext(type_ref);
                res.map(|ty_ns| (ty_ns, path.segments().first()))
            }),
            None => {
                let (ty, remaining_idx, _) = resolver.resolve_path_in_type_ns(db, path)?;
                match remaining_idx {
                    Some(remaining_idx) => {
                        if remaining_idx + 1 == path.segments().len() {
                            Some((ty, path.segments().last()))
                        } else {
                            None
                        }
                    }
                    None => Some((ty, None)),
                }
            }
        }?;

        // If we are in a TypeNs for a Trait, and we have an unresolved name, try to resolve it as a type
        // within the trait's associated types.
        if let (Some(unresolved), &TypeNs::TraitId(trait_id)) = (&unresolved, &ty)
            && let Some(type_alias_id) =
                trait_id.trait_items(db).associated_type_by_name(unresolved.name)
        {
            return Some(PathResolution::Def(ModuleDefId::from(type_alias_id).into()));
        }

        let res = match ty {
            TypeNs::SelfType(it) => PathResolution::SelfType(it.into()),
            TypeNs::GenericParam(id) => PathResolution::TypeParam(id.into()),
            TypeNs::AdtSelfType(it) | TypeNs::AdtId(it) => {
                PathResolution::Def(Adt::from(it).into())
            }
            TypeNs::EnumVariantId(it) => PathResolution::Def(Variant::from(it).into()),
            TypeNs::TypeAliasId(it) => PathResolution::Def(TypeAlias::from(it).into()),
            TypeNs::BuiltinType(it) => PathResolution::Def(BuiltinType::from(it).into()),
            TypeNs::TraitId(it) => PathResolution::Def(Trait::from(it).into()),
            TypeNs::TraitAliasId(it) => PathResolution::Def(TraitAlias::from(it).into()),
            TypeNs::ModuleId(it) => PathResolution::Def(ModuleDef::Module(it.into())),
        };
        match unresolved {
            Some(unresolved) => resolver
                .generic_def()
                .and_then(|def| {
                    hir_ty::associated_type_shorthand_candidates(
                        db,
                        def,
                        res.in_type_ns()?,
                        |name, id| (name == unresolved.name).then_some(id),
                    )
                })
                .map(TypeAlias::from)
                .map(Into::into)
                .map(PathResolution::Def),
            None => Some(res),
        }
    })()
    .or_else(|| {
        resolver
            .resolve_module_path_in_items(db, path.mod_path()?)
            .take_types()
            .map(|it| PathResolution::Def(it.into()))
    })
}

pub(crate) fn name_hygiene(db: &dyn HirDatabase, name: InFile<&SyntaxNode>) -> HygieneId {
    let Some(macro_file) = name.file_id.macro_file() else {
        return HygieneId::ROOT;
    };
    let span_map = db.expansion_span_map(macro_file);
    let ctx = span_map.span_at(name.value.text_range().start()).ctx;
    HygieneId::new(ctx.opaque_and_semitransparent(db))
}
