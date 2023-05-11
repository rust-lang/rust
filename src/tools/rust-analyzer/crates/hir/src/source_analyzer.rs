//! Lookup hir elements using positions in the source code. This is a lossy
//! transformation: in general, a single source might correspond to several
//! modules, functions, etc, due to macros, cfgs and `#[path=]` attributes on
//! modules.
//!
//! So, this modules should not be used during hir construction, it exists
//! purely for "IDE needs".
use std::{
    iter::{self, once},
    sync::Arc,
};

use either::Either;
use hir_def::{
    body::{
        self,
        scope::{ExprScopes, ScopeId},
        Body, BodySourceMap,
    },
    expr::{ExprId, Pat, PatId},
    lang_item::LangItem,
    macro_id_to_def_id,
    path::{ModPath, Path, PathKind},
    resolver::{resolver_for_scope, Resolver, TypeNs, ValueNs},
    type_ref::Mutability,
    AsMacroCall, AssocItemId, ConstId, DefWithBodyId, FieldId, FunctionId, ItemContainerId,
    LocalFieldId, Lookup, ModuleDefId, TraitId, VariantId,
};
use hir_expand::{
    builtin_fn_macro::BuiltinFnLikeExpander,
    hygiene::Hygiene,
    mod_path::path,
    name,
    name::{AsName, Name},
    HirFileId, InFile,
};
use hir_ty::{
    diagnostics::{
        record_literal_missing_fields, record_pattern_missing_fields, unsafe_expressions,
        UnsafeExpr,
    },
    method_resolution::{self, lang_items_for_bin_op},
    Adjustment, InferenceResult, Interner, Substitution, Ty, TyExt, TyKind, TyLoweringContext,
};
use itertools::Itertools;
use smallvec::SmallVec;
use syntax::{
    ast::{self, AstNode},
    SyntaxKind, SyntaxNode, TextRange, TextSize,
};

use crate::{
    db::HirDatabase, semantics::PathResolution, Adt, AssocItem, BindingMode, BuiltinAttr,
    BuiltinType, Callable, Const, DeriveHelper, Field, Function, Local, Macro, ModuleDef, Static,
    Struct, ToolModule, Trait, TraitAlias, Type, TypeAlias, Variant,
};

/// `SourceAnalyzer` is a convenience wrapper which exposes HIR API in terms of
/// original source files. It should not be used inside the HIR itself.
#[derive(Debug)]
pub(crate) struct SourceAnalyzer {
    pub(crate) file_id: HirFileId,
    pub(crate) resolver: Resolver,
    def: Option<(DefWithBodyId, Arc<Body>, Arc<BodySourceMap>)>,
    infer: Option<Arc<InferenceResult>>,
}

impl SourceAnalyzer {
    pub(crate) fn new_for_body(
        db: &dyn HirDatabase,
        def: DefWithBodyId,
        node @ InFile { file_id, .. }: InFile<&SyntaxNode>,
        offset: Option<TextSize>,
    ) -> SourceAnalyzer {
        let (body, source_map) = db.body_with_source_map(def);
        let scopes = db.expr_scopes(def);
        let scope = match offset {
            None => scope_for(&scopes, &source_map, node),
            Some(offset) => scope_for_offset(db, &scopes, &source_map, node.file_id, offset),
        };
        let resolver = resolver_for_scope(db.upcast(), def, scope);
        SourceAnalyzer {
            resolver,
            def: Some((def, body, source_map)),
            infer: Some(db.infer(def)),
            file_id,
        }
    }

    pub(crate) fn new_for_body_no_infer(
        db: &dyn HirDatabase,
        def: DefWithBodyId,
        node @ InFile { file_id, .. }: InFile<&SyntaxNode>,
        offset: Option<TextSize>,
    ) -> SourceAnalyzer {
        let (body, source_map) = db.body_with_source_map(def);
        let scopes = db.expr_scopes(def);
        let scope = match offset {
            None => scope_for(&scopes, &source_map, node),
            Some(offset) => scope_for_offset(db, &scopes, &source_map, node.file_id, offset),
        };
        let resolver = resolver_for_scope(db.upcast(), def, scope);
        SourceAnalyzer { resolver, def: Some((def, body, source_map)), infer: None, file_id }
    }

    pub(crate) fn new_for_resolver(
        resolver: Resolver,
        node: InFile<&SyntaxNode>,
    ) -> SourceAnalyzer {
        SourceAnalyzer { resolver, def: None, infer: None, file_id: node.file_id }
    }

    fn body_source_map(&self) -> Option<&BodySourceMap> {
        self.def.as_ref().map(|(.., source_map)| &**source_map)
    }
    fn body(&self) -> Option<&Body> {
        self.def.as_ref().map(|(_, body, _)| &**body)
    }

    fn expr_id(&self, db: &dyn HirDatabase, expr: &ast::Expr) -> Option<ExprId> {
        let src = match expr {
            ast::Expr::MacroExpr(expr) => {
                self.expand_expr(db, InFile::new(self.file_id, expr.macro_call()?))?
            }
            _ => InFile::new(self.file_id, expr.clone()),
        };
        let sm = self.body_source_map()?;
        sm.node_expr(src.as_ref())
    }

    fn pat_id(&self, pat: &ast::Pat) -> Option<PatId> {
        // FIXME: macros, see `expr_id`
        let src = InFile { file_id: self.file_id, value: pat };
        self.body_source_map()?.node_pat(src)
    }

    fn expand_expr(
        &self,
        db: &dyn HirDatabase,
        expr: InFile<ast::MacroCall>,
    ) -> Option<InFile<ast::Expr>> {
        let macro_file = self.body_source_map()?.node_macro_file(expr.as_ref())?;
        let expanded = db.parse_or_expand(macro_file)?;
        let res = if let Some(stmts) = ast::MacroStmts::cast(expanded.clone()) {
            match stmts.expr()? {
                ast::Expr::MacroExpr(mac) => {
                    self.expand_expr(db, InFile::new(macro_file, mac.macro_call()?))?
                }
                expr => InFile::new(macro_file, expr),
            }
        } else if let Some(call) = ast::MacroCall::cast(expanded.clone()) {
            self.expand_expr(db, InFile::new(macro_file, call))?
        } else {
            InFile::new(macro_file, ast::Expr::cast(expanded)?)
        };

        Some(res)
    }

    pub(crate) fn expr_adjustments(
        &self,
        db: &dyn HirDatabase,
        expr: &ast::Expr,
    ) -> Option<&[Adjustment]> {
        let expr_id = self.expr_id(db, expr)?;
        let infer = self.infer.as_ref()?;
        infer.expr_adjustments.get(&expr_id).map(|v| &**v)
    }

    pub(crate) fn type_of_expr(
        &self,
        db: &dyn HirDatabase,
        expr: &ast::Expr,
    ) -> Option<(Type, Option<Type>)> {
        let expr_id = self.expr_id(db, expr)?;
        let infer = self.infer.as_ref()?;
        let coerced = infer
            .expr_adjustments
            .get(&expr_id)
            .and_then(|adjusts| adjusts.last().map(|adjust| adjust.target.clone()));
        let ty = infer[expr_id].clone();
        let mk_ty = |ty| Type::new_with_resolver(db, &self.resolver, ty);
        Some((mk_ty(ty), coerced.map(mk_ty)))
    }

    pub(crate) fn type_of_pat(
        &self,
        db: &dyn HirDatabase,
        pat: &ast::Pat,
    ) -> Option<(Type, Option<Type>)> {
        let pat_id = self.pat_id(pat)?;
        let infer = self.infer.as_ref()?;
        let coerced = infer
            .pat_adjustments
            .get(&pat_id)
            .and_then(|adjusts| adjusts.last().map(|adjust| adjust.clone()));
        let ty = infer[pat_id].clone();
        let mk_ty = |ty| Type::new_with_resolver(db, &self.resolver, ty);
        Some((mk_ty(ty), coerced.map(mk_ty)))
    }

    pub(crate) fn type_of_self(
        &self,
        db: &dyn HirDatabase,
        param: &ast::SelfParam,
    ) -> Option<Type> {
        let src = InFile { file_id: self.file_id, value: param };
        let pat_id = self.body_source_map()?.node_self_param(src)?;
        let ty = self.infer.as_ref()?[pat_id].clone();
        Some(Type::new_with_resolver(db, &self.resolver, ty))
    }

    pub(crate) fn binding_mode_of_pat(
        &self,
        _db: &dyn HirDatabase,
        pat: &ast::IdentPat,
    ) -> Option<BindingMode> {
        let pat_id = self.pat_id(&pat.clone().into())?;
        let infer = self.infer.as_ref()?;
        infer.pat_binding_modes.get(&pat_id).map(|bm| match bm {
            hir_ty::BindingMode::Move => BindingMode::Move,
            hir_ty::BindingMode::Ref(hir_ty::Mutability::Mut) => BindingMode::Ref(Mutability::Mut),
            hir_ty::BindingMode::Ref(hir_ty::Mutability::Not) => {
                BindingMode::Ref(Mutability::Shared)
            }
        })
    }
    pub(crate) fn pattern_adjustments(
        &self,
        db: &dyn HirDatabase,
        pat: &ast::Pat,
    ) -> Option<SmallVec<[Type; 1]>> {
        let pat_id = self.pat_id(pat)?;
        let infer = self.infer.as_ref()?;
        Some(
            infer
                .pat_adjustments
                .get(&pat_id)?
                .iter()
                .map(|ty| Type::new_with_resolver(db, &self.resolver, ty.clone()))
                .collect(),
        )
    }

    pub(crate) fn resolve_method_call_as_callable(
        &self,
        db: &dyn HirDatabase,
        call: &ast::MethodCallExpr,
    ) -> Option<Callable> {
        let expr_id = self.expr_id(db, &call.clone().into())?;
        let (func, substs) = self.infer.as_ref()?.method_resolution(expr_id)?;
        let ty = db.value_ty(func.into()).substitute(Interner, &substs);
        let ty = Type::new_with_resolver(db, &self.resolver, ty);
        let mut res = ty.as_callable(db)?;
        res.is_bound_method = true;
        Some(res)
    }

    pub(crate) fn resolve_method_call(
        &self,
        db: &dyn HirDatabase,
        call: &ast::MethodCallExpr,
    ) -> Option<FunctionId> {
        let expr_id = self.expr_id(db, &call.clone().into())?;
        let (f_in_trait, substs) = self.infer.as_ref()?.method_resolution(expr_id)?;

        Some(self.resolve_impl_method_or_trait_def(db, f_in_trait, substs))
    }

    pub(crate) fn resolve_method_call_fallback(
        &self,
        db: &dyn HirDatabase,
        call: &ast::MethodCallExpr,
    ) -> Option<Either<FunctionId, FieldId>> {
        let expr_id = self.expr_id(db, &call.clone().into())?;
        let inference_result = self.infer.as_ref()?;
        match inference_result.method_resolution(expr_id) {
            Some((f_in_trait, substs)) => {
                Some(Either::Left(self.resolve_impl_method_or_trait_def(db, f_in_trait, substs)))
            }
            None => inference_result.field_resolution(expr_id).map(Either::Right),
        }
    }

    pub(crate) fn resolve_await_to_poll(
        &self,
        db: &dyn HirDatabase,
        await_expr: &ast::AwaitExpr,
    ) -> Option<FunctionId> {
        let mut ty = self.ty_of_expr(db, &await_expr.expr()?)?.clone();

        let into_future_trait = self
            .resolver
            .resolve_known_trait(db.upcast(), &path![core::future::IntoFuture])
            .map(Trait::from);

        if let Some(into_future_trait) = into_future_trait {
            let type_ = Type::new_with_resolver(db, &self.resolver, ty.clone());
            if type_.impls_trait(db, into_future_trait, &[]) {
                let items = into_future_trait.items(db);
                let into_future_type = items.into_iter().find_map(|item| match item {
                    AssocItem::TypeAlias(alias)
                        if alias.name(db) == hir_expand::name![IntoFuture] =>
                    {
                        Some(alias)
                    }
                    _ => None,
                })?;
                let future_trait = type_.normalize_trait_assoc_type(db, &[], into_future_type)?;
                ty = future_trait.ty;
            }
        }

        let future_trait = db.lang_item(self.resolver.krate(), LangItem::Future)?.as_trait()?;
        let poll_fn = db.lang_item(self.resolver.krate(), LangItem::FuturePoll)?.as_function()?;
        // HACK: subst for `poll()` coincides with that for `Future` because `poll()` itself
        // doesn't have any generic parameters, so we skip building another subst for `poll()`.
        let substs = hir_ty::TyBuilder::subst_for_def(db, future_trait, None).push(ty).build();
        Some(self.resolve_impl_method_or_trait_def(db, poll_fn, substs))
    }

    pub(crate) fn resolve_prefix_expr(
        &self,
        db: &dyn HirDatabase,
        prefix_expr: &ast::PrefixExpr,
    ) -> Option<FunctionId> {
        let (lang_item, fn_name) = match prefix_expr.op_kind()? {
            ast::UnaryOp::Deref => (LangItem::Deref, name![deref]),
            ast::UnaryOp::Not => (LangItem::Not, name![not]),
            ast::UnaryOp::Neg => (LangItem::Neg, name![neg]),
        };
        let ty = self.ty_of_expr(db, &prefix_expr.expr()?)?;

        let (op_trait, op_fn) = self.lang_trait_fn(db, lang_item, &fn_name)?;
        // HACK: subst for all methods coincides with that for their trait because the methods
        // don't have any generic parameters, so we skip building another subst for the methods.
        let substs = hir_ty::TyBuilder::subst_for_def(db, op_trait, None).push(ty.clone()).build();

        Some(self.resolve_impl_method_or_trait_def(db, op_fn, substs))
    }

    pub(crate) fn resolve_index_expr(
        &self,
        db: &dyn HirDatabase,
        index_expr: &ast::IndexExpr,
    ) -> Option<FunctionId> {
        let base_ty = self.ty_of_expr(db, &index_expr.base()?)?;
        let index_ty = self.ty_of_expr(db, &index_expr.index()?)?;

        let (op_trait, op_fn) = self.lang_trait_fn(db, LangItem::Index, &name![index])?;
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
        db: &dyn HirDatabase,
        binop_expr: &ast::BinExpr,
    ) -> Option<FunctionId> {
        let op = binop_expr.op_kind()?;
        let lhs = self.ty_of_expr(db, &binop_expr.lhs()?)?;
        let rhs = self.ty_of_expr(db, &binop_expr.rhs()?)?;

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
        db: &dyn HirDatabase,
        try_expr: &ast::TryExpr,
    ) -> Option<FunctionId> {
        let ty = self.ty_of_expr(db, &try_expr.expr()?)?;

        let op_fn = db.lang_item(self.resolver.krate(), LangItem::TryTraitBranch)?.as_function()?;
        let op_trait = match op_fn.lookup(db.upcast()).container {
            ItemContainerId::TraitId(id) => id,
            _ => return None,
        };
        // HACK: subst for `branch()` coincides with that for `Try` because `branch()` itself
        // doesn't have any generic parameters, so we skip building another subst for `branch()`.
        let substs = hir_ty::TyBuilder::subst_for_def(db, op_trait, None).push(ty.clone()).build();

        Some(self.resolve_impl_method_or_trait_def(db, op_fn, substs))
    }

    pub(crate) fn resolve_field(
        &self,
        db: &dyn HirDatabase,
        field: &ast::FieldExpr,
    ) -> Option<Field> {
        let expr_id = self.expr_id(db, &field.clone().into())?;
        self.infer.as_ref()?.field_resolution(expr_id).map(|it| it.into())
    }

    pub(crate) fn resolve_record_field(
        &self,
        db: &dyn HirDatabase,
        field: &ast::RecordExprField,
    ) -> Option<(Field, Option<Local>, Type)> {
        let record_expr = ast::RecordExpr::cast(field.syntax().parent().and_then(|p| p.parent())?)?;
        let expr = ast::Expr::from(record_expr);
        let expr_id = self.body_source_map()?.node_expr(InFile::new(self.file_id, &expr))?;

        let local_name = field.field_name()?.as_name();
        let local = if field.name_ref().is_some() {
            None
        } else {
            // Shorthand syntax, resolve to the local
            let path = ModPath::from_segments(PathKind::Plain, once(local_name.clone()));
            match self.resolver.resolve_path_in_value_ns_fully(db.upcast(), &path) {
                Some(ValueNs::LocalBinding(binding_id)) => {
                    Some(Local { binding_id, parent: self.resolver.body_owner()? })
                }
                _ => None,
            }
        };
        let (_, subst) = self.infer.as_ref()?.type_of_expr.get(expr_id)?.as_adt()?;
        let variant = self.infer.as_ref()?.variant_resolution_for_expr(expr_id)?;
        let variant_data = variant.variant_data(db.upcast());
        let field = FieldId { parent: variant, local_id: variant_data.field(&local_name)? };
        let field_ty =
            db.field_types(variant).get(field.local_id)?.clone().substitute(Interner, subst);
        Some((field.into(), local, Type::new_with_resolver(db, &self.resolver, field_ty)))
    }

    pub(crate) fn resolve_record_pat_field(
        &self,
        db: &dyn HirDatabase,
        field: &ast::RecordPatField,
    ) -> Option<(Field, Type)> {
        let field_name = field.field_name()?.as_name();
        let record_pat = ast::RecordPat::cast(field.syntax().parent().and_then(|p| p.parent())?)?;
        let pat_id = self.pat_id(&record_pat.into())?;
        let variant = self.infer.as_ref()?.variant_resolution_for_pat(pat_id)?;
        let variant_data = variant.variant_data(db.upcast());
        let field = FieldId { parent: variant, local_id: variant_data.field(&field_name)? };
        let (_, subst) = self.infer.as_ref()?.type_of_pat.get(pat_id)?.as_adt()?;
        let field_ty =
            db.field_types(variant).get(field.local_id)?.clone().substitute(Interner, subst);
        Some((field.into(), Type::new_with_resolver(db, &self.resolver, field_ty)))
    }

    pub(crate) fn resolve_macro_call(
        &self,
        db: &dyn HirDatabase,
        macro_call: InFile<&ast::MacroCall>,
    ) -> Option<Macro> {
        let ctx = body::LowerCtx::new(db.upcast(), macro_call.file_id);
        let path = macro_call.value.path().and_then(|ast| Path::from_src(ast, &ctx))?;
        self.resolver.resolve_path_as_macro(db.upcast(), path.mod_path()).map(|it| it.into())
    }

    pub(crate) fn resolve_bind_pat_to_const(
        &self,
        db: &dyn HirDatabase,
        pat: &ast::IdentPat,
    ) -> Option<ModuleDef> {
        let pat_id = self.pat_id(&pat.clone().into())?;
        let body = self.body()?;
        let path = match &body[pat_id] {
            Pat::Path(path) => path,
            _ => return None,
        };
        let res = resolve_hir_path(db, &self.resolver, path)?;
        match res {
            PathResolution::Def(def) => Some(def),
            _ => None,
        }
    }

    pub(crate) fn resolve_path(
        &self,
        db: &dyn HirDatabase,
        path: &ast::Path,
    ) -> Option<PathResolution> {
        let parent = path.syntax().parent();
        let parent = || parent.clone();

        let mut prefer_value_ns = false;
        let resolved = (|| {
            let infer = self.infer.as_deref()?;
            if let Some(path_expr) = parent().and_then(ast::PathExpr::cast) {
                let expr_id = self.expr_id(db, &path_expr.into())?;
                if let Some((assoc, subs)) = infer.assoc_resolutions_for_expr(expr_id) {
                    let assoc = match assoc {
                        AssocItemId::FunctionId(f_in_trait) => {
                            match infer.type_of_expr.get(expr_id) {
                                None => assoc,
                                Some(func_ty) => {
                                    if let TyKind::FnDef(_fn_def, subs) = func_ty.kind(Interner) {
                                        self.resolve_impl_method_or_trait_def(
                                            db,
                                            f_in_trait,
                                            subs.clone(),
                                        )
                                        .into()
                                    } else {
                                        assoc
                                    }
                                }
                            }
                        }
                        AssocItemId::ConstId(const_id) => {
                            self.resolve_impl_const_or_trait_def(db, const_id, subs).into()
                        }
                        assoc => assoc,
                    };

                    return Some(PathResolution::Def(AssocItem::from(assoc).into()));
                }
                if let Some(VariantId::EnumVariantId(variant)) =
                    infer.variant_resolution_for_expr(expr_id)
                {
                    return Some(PathResolution::Def(ModuleDef::Variant(variant.into())));
                }
                prefer_value_ns = true;
            } else if let Some(path_pat) = parent().and_then(ast::PathPat::cast) {
                let pat_id = self.pat_id(&path_pat.into())?;
                if let Some((assoc, subs)) = infer.assoc_resolutions_for_pat(pat_id) {
                    let assoc = match assoc {
                        AssocItemId::ConstId(const_id) => {
                            self.resolve_impl_const_or_trait_def(db, const_id, subs).into()
                        }
                        assoc => assoc,
                    };
                    return Some(PathResolution::Def(AssocItem::from(assoc).into()));
                }
                if let Some(VariantId::EnumVariantId(variant)) =
                    infer.variant_resolution_for_pat(pat_id)
                {
                    return Some(PathResolution::Def(ModuleDef::Variant(variant.into())));
                }
            } else if let Some(rec_lit) = parent().and_then(ast::RecordExpr::cast) {
                let expr_id = self.expr_id(db, &rec_lit.into())?;
                if let Some(VariantId::EnumVariantId(variant)) =
                    infer.variant_resolution_for_expr(expr_id)
                {
                    return Some(PathResolution::Def(ModuleDef::Variant(variant.into())));
                }
            } else {
                let record_pat = parent().and_then(ast::RecordPat::cast).map(ast::Pat::from);
                let tuple_struct_pat =
                    || parent().and_then(ast::TupleStructPat::cast).map(ast::Pat::from);
                if let Some(pat) = record_pat.or_else(tuple_struct_pat) {
                    let pat_id = self.pat_id(&pat)?;
                    let variant_res_for_pat = infer.variant_resolution_for_pat(pat_id);
                    if let Some(VariantId::EnumVariantId(variant)) = variant_res_for_pat {
                        return Some(PathResolution::Def(ModuleDef::Variant(variant.into())));
                    }
                }
            }
            None
        })();
        if let Some(_) = resolved {
            return resolved;
        }

        // This must be a normal source file rather than macro file.
        let hygiene = Hygiene::new(db.upcast(), self.file_id);
        let ctx = body::LowerCtx::with_hygiene(db.upcast(), &hygiene);
        let hir_path = Path::from_src(path.clone(), &ctx)?;

        // Case where path is a qualifier of a use tree, e.g. foo::bar::{Baz, Qux} where we are
        // trying to resolve foo::bar.
        if let Some(use_tree) = parent().and_then(ast::UseTree::cast) {
            if use_tree.coloncolon_token().is_some() {
                return resolve_hir_path_qualifier(db, &self.resolver, &hir_path);
            }
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
        if path.parent_path().is_some() {
            return match resolve_hir_path_qualifier(db, &self.resolver, &hir_path) {
                None if meta_path.is_some() => {
                    path.first_segment().and_then(|it| it.name_ref()).and_then(|name_ref| {
                        ToolModule::by_name(db, self.resolver.krate().into(), &name_ref.text())
                            .map(PathResolution::ToolModule)
                    })
                }
                res => res,
            };
        } else if let Some(meta_path) = meta_path {
            // Case where we are resolving the final path segment of a path in an attribute
            // in this case we have to check for inert/builtin attributes and tools and prioritize
            // resolution of attributes over other namespaces
            if let Some(name_ref) = path.as_single_name_ref() {
                let builtin =
                    BuiltinAttr::by_name(db, self.resolver.krate().into(), &name_ref.text());
                if let Some(_) = builtin {
                    return builtin.map(PathResolution::BuiltinAttr);
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
                                helpers.iter().group_by(|(_, macro_id, ..)| macro_id).into_iter()
                            {
                                if let Some(idx) = helpers.position(|(name, ..)| *name == name_ref)
                                {
                                    return Some(PathResolution::DeriveHelper(DeriveHelper {
                                        derive: *macro_id,
                                        idx: idx as u32,
                                    }));
                                }
                            }
                        }
                    }
                }
            }
            return match resolve_hir_path_as_macro(db, &self.resolver, &hir_path) {
                Some(m) => Some(PathResolution::Def(ModuleDef::Macro(m))),
                // this labels any path that starts with a tool module as the tool itself, this is technically wrong
                // but there is no benefit in differentiating these two cases for the time being
                None => path.first_segment().and_then(|it| it.name_ref()).and_then(|name_ref| {
                    ToolModule::by_name(db, self.resolver.krate().into(), &name_ref.text())
                        .map(PathResolution::ToolModule)
                }),
            };
        }
        if parent().map_or(false, |it| ast::Visibility::can_cast(it.kind())) {
            resolve_hir_path_qualifier(db, &self.resolver, &hir_path)
        } else {
            resolve_hir_path_(db, &self.resolver, &hir_path, prefer_value_ns)
        }
    }

    pub(crate) fn record_literal_missing_fields(
        &self,
        db: &dyn HirDatabase,
        literal: &ast::RecordExpr,
    ) -> Option<Vec<(Field, Type)>> {
        let body = self.body()?;
        let infer = self.infer.as_ref()?;

        let expr_id = self.expr_id(db, &literal.clone().into())?;
        let substs = infer.type_of_expr[expr_id].as_adt()?.1;

        let (variant, missing_fields, _exhaustive) =
            record_literal_missing_fields(db, infer, expr_id, &body[expr_id])?;
        let res = self.missing_fields(db, substs, variant, missing_fields);
        Some(res)
    }

    pub(crate) fn record_pattern_missing_fields(
        &self,
        db: &dyn HirDatabase,
        pattern: &ast::RecordPat,
    ) -> Option<Vec<(Field, Type)>> {
        let body = self.body()?;
        let infer = self.infer.as_ref()?;

        let pat_id = self.pat_id(&pattern.clone().into())?;
        let substs = infer.type_of_pat[pat_id].as_adt()?.1;

        let (variant, missing_fields, _exhaustive) =
            record_pattern_missing_fields(db, infer, pat_id, &body[pat_id])?;
        let res = self.missing_fields(db, substs, variant, missing_fields);
        Some(res)
    }

    fn missing_fields(
        &self,
        db: &dyn HirDatabase,
        substs: &Substitution,
        variant: VariantId,
        missing_fields: Vec<LocalFieldId>,
    ) -> Vec<(Field, Type)> {
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

    pub(crate) fn expand(
        &self,
        db: &dyn HirDatabase,
        macro_call: InFile<&ast::MacroCall>,
    ) -> Option<HirFileId> {
        let krate = self.resolver.krate();
        let macro_call_id = macro_call.as_call_id(db.upcast(), krate, |path| {
            self.resolver
                .resolve_path_as_macro(db.upcast(), &path)
                .map(|it| macro_id_to_def_id(db.upcast(), it))
        })?;
        Some(macro_call_id.as_file()).filter(|it| it.expansion_level(db.upcast()) < 64)
    }

    pub(crate) fn resolve_variant(
        &self,
        db: &dyn HirDatabase,
        record_lit: ast::RecordExpr,
    ) -> Option<VariantId> {
        let infer = self.infer.as_ref()?;
        let expr_id = self.expr_id(db, &record_lit.into())?;
        infer.variant_resolution_for_expr(expr_id)
    }

    pub(crate) fn is_unsafe_macro_call(
        &self,
        db: &dyn HirDatabase,
        macro_call: InFile<&ast::MacroCall>,
    ) -> bool {
        // check for asm/global_asm
        if let Some(mac) = self.resolve_macro_call(db, macro_call) {
            let ex = match mac.id {
                hir_def::MacroId::Macro2Id(it) => it.lookup(db.upcast()).expander,
                hir_def::MacroId::MacroRulesId(it) => it.lookup(db.upcast()).expander,
                _ => hir_def::MacroExpander::Declarative,
            };
            match ex {
                hir_def::MacroExpander::BuiltIn(e)
                    if e == BuiltinFnLikeExpander::Asm || e == BuiltinFnLikeExpander::GlobalAsm =>
                {
                    return true
                }
                _ => (),
            }
        }
        let macro_expr = match macro_call
            .map(|it| it.syntax().parent().and_then(ast::MacroExpr::cast))
            .transpose()
        {
            Some(it) => it,
            None => return false,
        };

        if let (Some((def, body, sm)), Some(infer)) = (&self.def, &self.infer) {
            if let Some(expanded_expr) = sm.macro_expansion_expr(macro_expr.as_ref()) {
                let mut is_unsafe = false;
                unsafe_expressions(
                    db,
                    infer,
                    *def,
                    body,
                    expanded_expr,
                    &mut |UnsafeExpr { inside_unsafe_block, .. }| is_unsafe |= !inside_unsafe_block,
                );
                return is_unsafe;
            }
        }
        false
    }

    fn resolve_impl_method_or_trait_def(
        &self,
        db: &dyn HirDatabase,
        func: FunctionId,
        substs: Substitution,
    ) -> FunctionId {
        let krate = self.resolver.krate();
        let owner = match self.resolver.body_owner() {
            Some(it) => it,
            None => return func,
        };
        let env = owner.as_generic_def_id().map_or_else(
            || Arc::new(hir_ty::TraitEnvironment::empty(krate)),
            |d| db.trait_environment(d),
        );
        method_resolution::lookup_impl_method(db, env, func, substs).0
    }

    fn resolve_impl_const_or_trait_def(
        &self,
        db: &dyn HirDatabase,
        const_id: ConstId,
        subs: Substitution,
    ) -> ConstId {
        let krate = self.resolver.krate();
        let owner = match self.resolver.body_owner() {
            Some(it) => it,
            None => return const_id,
        };
        let env = owner.as_generic_def_id().map_or_else(
            || Arc::new(hir_ty::TraitEnvironment::empty(krate)),
            |d| db.trait_environment(d),
        );
        method_resolution::lookup_impl_const(db, env, const_id, subs).0
    }

    fn lang_trait_fn(
        &self,
        db: &dyn HirDatabase,
        lang_trait: LangItem,
        method_name: &Name,
    ) -> Option<(TraitId, FunctionId)> {
        let trait_id = db.lang_item(self.resolver.krate(), lang_trait)?.as_trait()?;
        let fn_id = db.trait_data(trait_id).method_by_name(method_name)?;
        Some((trait_id, fn_id))
    }

    fn ty_of_expr(&self, db: &dyn HirDatabase, expr: &ast::Expr) -> Option<&Ty> {
        self.infer.as_ref()?.type_of_expr.get(self.expr_id(db, expr)?)
    }
}

fn scope_for(
    scopes: &ExprScopes,
    source_map: &BodySourceMap,
    node: InFile<&SyntaxNode>,
) -> Option<ScopeId> {
    node.value
        .ancestors()
        .filter_map(ast::Expr::cast)
        .filter_map(|it| source_map.node_expr(InFile::new(node.file_id, &it)))
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
            let InFile { file_id, value } = source_map.expr_syntax(*id).ok()?;
            if from_file == file_id {
                return Some((value.text_range(), scope));
            }

            // FIXME handle attribute expansion
            let source = iter::successors(file_id.call_node(db.upcast()), |it| {
                it.file_id.call_node(db.upcast())
            })
            .find(|it| it.file_id == from_file)
            .filter(|it| it.value.kind() == SyntaxKind::MACRO_CALL)?;
            Some((source.value.text_range(), scope))
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
            let source = source_map.expr_syntax(*id).ok()?;
            // FIXME: correctly handle macro expansion
            if source.file_id != from_file {
                return None;
            }
            let root = source.file_syntax(db.upcast());
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
    resolver: &Resolver,
    path: &Path,
) -> Option<PathResolution> {
    resolve_hir_path_(db, resolver, path, false)
}

#[inline]
pub(crate) fn resolve_hir_path_as_macro(
    db: &dyn HirDatabase,
    resolver: &Resolver,
    path: &Path,
) -> Option<Macro> {
    resolver.resolve_path_as_macro(db.upcast(), path.mod_path()).map(Into::into)
}

fn resolve_hir_path_(
    db: &dyn HirDatabase,
    resolver: &Resolver,
    path: &Path,
    prefer_value_ns: bool,
) -> Option<PathResolution> {
    let types = || {
        let (ty, unresolved) = match path.type_anchor() {
            Some(type_ref) => {
                let (_, res) = TyLoweringContext::new(db, resolver).lower_ty_ext(type_ref);
                res.map(|ty_ns| (ty_ns, path.segments().first()))
            }
            None => {
                let (ty, remaining_idx) =
                    resolver.resolve_path_in_type_ns(db.upcast(), path.mod_path())?;
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
        if let (Some(unresolved), &TypeNs::TraitId(trait_id)) = (&unresolved, &ty) {
            if let Some(type_alias_id) =
                db.trait_data(trait_id).associated_type_by_name(unresolved.name)
            {
                return Some(PathResolution::Def(ModuleDefId::from(type_alias_id).into()));
            }
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
    let values = || {
        resolver.resolve_path_in_value_ns_fully(db.upcast(), path.mod_path()).and_then(|val| {
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
    };

    let items = || {
        resolver
            .resolve_module_path_in_items(db.upcast(), path.mod_path())
            .take_types()
            .map(|it| PathResolution::Def(it.into()))
    };

    let macros = || {
        resolver
            .resolve_path_as_macro(db.upcast(), path.mod_path())
            .map(|def| PathResolution::Def(ModuleDef::Macro(def.into())))
    };

    if prefer_value_ns { values().or_else(types) } else { types().or_else(values) }
        .or_else(items)
        .or_else(macros)
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
    resolver: &Resolver,
    path: &Path,
) -> Option<PathResolution> {
    resolver
        .resolve_path_in_type_ns_fully(db.upcast(), path.mod_path())
        .map(|ty| match ty {
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
        })
        .or_else(|| {
            resolver
                .resolve_module_path_in_items(db.upcast(), path.mod_path())
                .take_types()
                .map(|it| PathResolution::Def(it.into()))
        })
}
