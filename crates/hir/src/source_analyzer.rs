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

use hir_def::{
    body::{
        self,
        scope::{ExprScopes, ScopeId},
        Body, BodySourceMap,
    },
    expr::{ExprId, Pat, PatId},
    macro_id_to_def_id,
    path::{ModPath, Path, PathKind},
    resolver::{resolver_for_scope, Resolver, TypeNs, ValueNs},
    AsMacroCall, DefWithBodyId, FieldId, FunctionId, LocalFieldId, ModuleDefId, VariantId,
};
use hir_expand::{hygiene::Hygiene, name::AsName, HirFileId, InFile};
use hir_ty::{
    diagnostics::{record_literal_missing_fields, record_pattern_missing_fields},
    InferenceResult, Interner, Substitution, TyExt, TyLoweringContext,
};
use syntax::{
    ast::{self, AstNode},
    SyntaxKind, SyntaxNode, TextRange, TextSize,
};

use crate::{
    db::HirDatabase, semantics::PathResolution, Adt, BuiltinAttr, BuiltinType, Const, Field,
    Function, Local, Macro, ModuleDef, Static, Struct, ToolModule, Trait, Type, TypeAlias, Variant,
};
use base_db::CrateId;

/// `SourceAnalyzer` is a convenience wrapper which exposes HIR API in terms of
/// original source files. It should not be used inside the HIR itself.
#[derive(Debug)]
pub(crate) struct SourceAnalyzer {
    pub(crate) file_id: HirFileId,
    pub(crate) resolver: Resolver,
    body: Option<Arc<Body>>,
    body_source_map: Option<Arc<BodySourceMap>>,
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
            Some(offset) => scope_for_offset(db, &scopes, &source_map, node.with_value(offset)),
        };
        let resolver = resolver_for_scope(db.upcast(), def, scope);
        SourceAnalyzer {
            resolver,
            body: Some(body),
            body_source_map: Some(source_map),
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
            Some(offset) => scope_for_offset(db, &scopes, &source_map, node.with_value(offset)),
        };
        let resolver = resolver_for_scope(db.upcast(), def, scope);
        SourceAnalyzer {
            resolver,
            body: Some(body),
            body_source_map: Some(source_map),
            infer: None,
            file_id,
        }
    }

    pub(crate) fn new_for_resolver(
        resolver: Resolver,
        node: InFile<&SyntaxNode>,
    ) -> SourceAnalyzer {
        SourceAnalyzer {
            resolver,
            body: None,
            body_source_map: None,
            infer: None,
            file_id: node.file_id,
        }
    }

    fn expr_id(&self, db: &dyn HirDatabase, expr: &ast::Expr) -> Option<ExprId> {
        let src = match expr {
            ast::Expr::MacroCall(call) => {
                self.expand_expr(db, InFile::new(self.file_id, call.clone()))?
            }
            _ => InFile::new(self.file_id, expr.clone()),
        };
        let sm = self.body_source_map.as_ref()?;
        sm.node_expr(src.as_ref())
    }

    fn pat_id(&self, pat: &ast::Pat) -> Option<PatId> {
        // FIXME: macros, see `expr_id`
        let src = InFile { file_id: self.file_id, value: pat };
        self.body_source_map.as_ref()?.node_pat(src)
    }

    fn expand_expr(
        &self,
        db: &dyn HirDatabase,
        expr: InFile<ast::MacroCall>,
    ) -> Option<InFile<ast::Expr>> {
        let macro_file = self.body_source_map.as_ref()?.node_macro_file(expr.as_ref())?;
        let expanded = db.parse_or_expand(macro_file)?;

        let res = match ast::MacroCall::cast(expanded.clone()) {
            Some(call) => self.expand_expr(db, InFile::new(macro_file, call))?,
            _ => InFile::new(macro_file, ast::Expr::cast(expanded)?),
        };
        Some(res)
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
        mk_ty(ty).zip(Some(coerced.and_then(mk_ty)))
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
            .and_then(|adjusts| adjusts.last().map(|adjust| adjust.target.clone()));
        let ty = infer[pat_id].clone();
        let mk_ty = |ty| Type::new_with_resolver(db, &self.resolver, ty);
        mk_ty(ty).zip(Some(coerced.and_then(mk_ty)))
    }

    pub(crate) fn type_of_self(
        &self,
        db: &dyn HirDatabase,
        param: &ast::SelfParam,
    ) -> Option<Type> {
        let src = InFile { file_id: self.file_id, value: param };
        let pat_id = self.body_source_map.as_ref()?.node_self_param(src)?;
        let ty = self.infer.as_ref()?[pat_id].clone();
        Type::new_with_resolver(db, &self.resolver, ty)
    }

    pub(crate) fn resolve_method_call(
        &self,
        db: &dyn HirDatabase,
        call: &ast::MethodCallExpr,
    ) -> Option<(FunctionId, Substitution)> {
        let expr_id = self.expr_id(db, &call.clone().into())?;
        self.infer.as_ref()?.method_resolution(expr_id)
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
        let expr_id = self.body_source_map.as_ref()?.node_expr(InFile::new(self.file_id, &expr))?;

        let local_name = field.field_name()?.as_name();
        let local = if field.name_ref().is_some() {
            None
        } else {
            let path = ModPath::from_segments(PathKind::Plain, once(local_name.clone()));
            match self.resolver.resolve_path_in_value_ns_fully(db.upcast(), &path) {
                Some(ValueNs::LocalBinding(pat_id)) => {
                    Some(Local { pat_id, parent: self.resolver.body_owner()? })
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
        Some((field.into(), local, Type::new_with_resolver(db, &self.resolver, field_ty)?))
    }

    pub(crate) fn resolve_record_pat_field(
        &self,
        db: &dyn HirDatabase,
        field: &ast::RecordPatField,
    ) -> Option<Field> {
        let field_name = field.field_name()?.as_name();
        let record_pat = ast::RecordPat::cast(field.syntax().parent().and_then(|p| p.parent())?)?;
        let pat_id = self.pat_id(&record_pat.into())?;
        let variant = self.infer.as_ref()?.variant_resolution_for_pat(pat_id)?;
        let variant_data = variant.variant_data(db.upcast());
        let field = FieldId { parent: variant, local_id: variant_data.field(&field_name)? };
        Some(field.into())
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
        let body = self.body.as_ref()?;
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
        if let Some(path_expr) = parent().and_then(ast::PathExpr::cast) {
            let expr_id = self.expr_id(db, &path_expr.into())?;
            let infer = self.infer.as_ref()?;
            if let Some(assoc) = infer.assoc_resolutions_for_expr(expr_id) {
                return Some(PathResolution::AssocItem(assoc.into()));
            }
            if let Some(VariantId::EnumVariantId(variant)) =
                infer.variant_resolution_for_expr(expr_id)
            {
                return Some(PathResolution::Def(ModuleDef::Variant(variant.into())));
            }
            prefer_value_ns = true;
        } else if let Some(path_pat) = parent().and_then(ast::PathPat::cast) {
            let pat_id = self.pat_id(&path_pat.into())?;
            if let Some(assoc) = self.infer.as_ref()?.assoc_resolutions_for_pat(pat_id) {
                return Some(PathResolution::AssocItem(assoc.into()));
            }
            if let Some(VariantId::EnumVariantId(variant)) =
                self.infer.as_ref()?.variant_resolution_for_pat(pat_id)
            {
                return Some(PathResolution::Def(ModuleDef::Variant(variant.into())));
            }
        } else if let Some(rec_lit) = parent().and_then(ast::RecordExpr::cast) {
            let expr_id = self.expr_id(db, &rec_lit.into())?;
            if let Some(VariantId::EnumVariantId(variant)) =
                self.infer.as_ref()?.variant_resolution_for_expr(expr_id)
            {
                return Some(PathResolution::Def(ModuleDef::Variant(variant.into())));
            }
        }

        let record_pat = parent().and_then(ast::RecordPat::cast).map(ast::Pat::from);
        let tuple_struct_pat = || parent().and_then(ast::TupleStructPat::cast).map(ast::Pat::from);
        if let Some(pat) = record_pat.or_else(tuple_struct_pat) {
            let pat_id = self.pat_id(&pat)?;
            let variant_res_for_pat = self.infer.as_ref()?.variant_resolution_for_pat(pat_id);
            if let Some(VariantId::EnumVariantId(variant)) = variant_res_for_pat {
                return Some(PathResolution::Def(ModuleDef::Variant(variant.into())));
            }
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

        let is_path_of_attr = path
            .syntax()
            .ancestors()
            .map(|it| it.kind())
            .take_while(|&kind| ast::Path::can_cast(kind) || ast::Meta::can_cast(kind))
            .last()
            .map_or(false, ast::Meta::can_cast);

        // Case where path is a qualifier of another path, e.g. foo::bar::Baz where we are
        // trying to resolve foo::bar.
        if path.parent_path().is_some() {
            return match resolve_hir_path_qualifier(db, &self.resolver, &hir_path) {
                None if is_path_of_attr => {
                    path.first_segment().and_then(|it| it.name_ref()).and_then(|name_ref| {
                        match self.resolver.krate() {
                            Some(krate) => ToolModule::by_name(db, krate.into(), &name_ref.text()),
                            None => ToolModule::builtin(&name_ref.text()),
                        }
                        .map(PathResolution::ToolModule)
                    })
                }
                res => res,
            };
        } else if is_path_of_attr {
            // Case where we are resolving the final path segment of a path in an attribute
            // in this case we have to check for inert/builtin attributes and tools and prioritize
            // resolution of attributes over other namespaces
            let name_ref = path.as_single_name_ref();
            let builtin = name_ref.as_ref().and_then(|name_ref| match self.resolver.krate() {
                Some(krate) => BuiltinAttr::by_name(db, krate.into(), &name_ref.text()),
                None => BuiltinAttr::builtin(&name_ref.text()),
            });
            if let builtin @ Some(_) = builtin {
                return builtin.map(PathResolution::BuiltinAttr);
            }
            return match resolve_hir_path_as_macro(db, &self.resolver, &hir_path) {
                Some(m) => Some(PathResolution::Def(ModuleDef::Macro(m))),
                // this labels any path that starts with a tool module as the tool itself, this is technically wrong
                // but there is no benefit in differentiating these two cases for the time being
                None => path.first_segment().and_then(|it| it.name_ref()).and_then(|name_ref| {
                    match self.resolver.krate() {
                        Some(krate) => ToolModule::by_name(db, krate.into(), &name_ref.text()),
                        None => ToolModule::builtin(&name_ref.text()),
                    }
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
        let krate = self.resolver.krate()?;
        let body = self.body.as_ref()?;
        let infer = self.infer.as_ref()?;

        let expr_id = self.expr_id(db, &literal.clone().into())?;
        let substs = infer.type_of_expr[expr_id].as_adt()?.1;

        let (variant, missing_fields, _exhaustive) =
            record_literal_missing_fields(db, infer, expr_id, &body[expr_id])?;
        let res = self.missing_fields(db, krate, substs, variant, missing_fields);
        Some(res)
    }

    pub(crate) fn record_pattern_missing_fields(
        &self,
        db: &dyn HirDatabase,
        pattern: &ast::RecordPat,
    ) -> Option<Vec<(Field, Type)>> {
        let krate = self.resolver.krate()?;
        let body = self.body.as_ref()?;
        let infer = self.infer.as_ref()?;

        let pat_id = self.pat_id(&pattern.clone().into())?;
        let substs = infer.type_of_pat[pat_id].as_adt()?.1;

        let (variant, missing_fields, _exhaustive) =
            record_pattern_missing_fields(db, infer, pat_id, &body[pat_id])?;
        let res = self.missing_fields(db, krate, substs, variant, missing_fields);
        Some(res)
    }

    fn missing_fields(
        &self,
        db: &dyn HirDatabase,
        krate: CrateId,
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
                (field.into(), Type::new_with_resolver_inner(db, krate, &self.resolver, ty))
            })
            .collect()
    }

    pub(crate) fn expand(
        &self,
        db: &dyn HirDatabase,
        macro_call: InFile<&ast::MacroCall>,
    ) -> Option<HirFileId> {
        let krate = self.resolver.krate()?;
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
    offset: InFile<TextSize>,
) -> Option<ScopeId> {
    scopes
        .scope_by_expr()
        .iter()
        .filter_map(|(id, scope)| {
            let InFile { file_id, value } = source_map.expr_syntax(*id).ok()?;
            if offset.file_id == file_id {
                let root = db.parse_or_expand(file_id)?;
                let node = value.to_node(&root);
                return Some((node.syntax().text_range(), scope));
            }

            // FIXME handle attribute expansion
            let source = iter::successors(file_id.call_node(db.upcast()), |it| {
                it.file_id.call_node(db.upcast())
            })
            .find(|it| it.file_id == offset.file_id)
            .filter(|it| it.value.kind() == SyntaxKind::MACRO_CALL)?;
            Some((source.value.text_range(), scope))
        })
        // find containing scope
        .min_by_key(|(expr_range, _scope)| {
            (
                !(expr_range.start() <= offset.value && offset.value <= expr_range.end()),
                expr_range.len(),
            )
        })
        .map(|(expr_range, scope)| {
            adjust(db, scopes, source_map, expr_range, offset).unwrap_or(*scope)
        })
}

// XXX: during completion, cursor might be outside of any particular
// expression. Try to figure out the correct scope...
fn adjust(
    db: &dyn HirDatabase,
    scopes: &ExprScopes,
    source_map: &BodySourceMap,
    expr_range: TextRange,
    offset: InFile<TextSize>,
) -> Option<ScopeId> {
    let child_scopes = scopes
        .scope_by_expr()
        .iter()
        .filter_map(|(id, scope)| {
            let source = source_map.expr_syntax(*id).ok()?;
            // FIXME: correctly handle macro expansion
            if source.file_id != offset.file_id {
                return None;
            }
            let root = source.file_syntax(db.upcast());
            let node = source.value.to_node(&root);
            Some((node.syntax().text_range(), scope))
        })
        .filter(|&(range, _)| {
            range.start() <= offset.value && expr_range.contains_range(range) && range != expr_range
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
                let (ty, remaining) =
                    resolver.resolve_path_in_type_ns(db.upcast(), path.mod_path())?;
                match remaining {
                    Some(remaining) if remaining > 1 => None,
                    _ => Some((ty, path.segments().get(1))),
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
        };
        match unresolved {
            Some(unresolved) => resolver
                .generic_def()
                .and_then(|def| {
                    hir_ty::associated_type_shorthand_candidates(
                        db,
                        def,
                        res.in_type_ns()?,
                        |name, _, id| (name == unresolved.name).then(|| id),
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
                ValueNs::LocalBinding(pat_id) => {
                    let var = Local { parent: body_owner?, pat_id };
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
    let items = resolver
        .resolve_module_path_in_items(db.upcast(), path.mod_path())
        .take_types()
        .map(|it| PathResolution::Def(it.into()));

    if items.is_some() {
        return items;
    }

    resolver.resolve_path_in_type_ns_fully(db.upcast(), path.mod_path()).map(|ty| match ty {
        TypeNs::SelfType(it) => PathResolution::SelfType(it.into()),
        TypeNs::GenericParam(id) => PathResolution::TypeParam(id.into()),
        TypeNs::AdtSelfType(it) | TypeNs::AdtId(it) => PathResolution::Def(Adt::from(it).into()),
        TypeNs::EnumVariantId(it) => PathResolution::Def(Variant::from(it).into()),
        TypeNs::TypeAliasId(it) => PathResolution::Def(TypeAlias::from(it).into()),
        TypeNs::BuiltinType(it) => PathResolution::Def(BuiltinType::from(it).into()),
        TypeNs::TraitId(it) => PathResolution::Def(Trait::from(it).into()),
    })
}
