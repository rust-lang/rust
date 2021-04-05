//! Lookup hir elements using positions in the source code. This is a lossy
//! transformation: in general, a single source might correspond to several
//! modules, functions, etc, due to macros, cfgs and `#[path=]` attributes on
//! modules.
//!
//! So, this modules should not be used during hir construction, it exists
//! purely for "IDE needs".
use std::{iter::once, sync::Arc};

use hir_def::{
    body::{
        scope::{ExprScopes, ScopeId},
        Body, BodySourceMap,
    },
    expr::{ExprId, Pat, PatId},
    path::{ModPath, Path, PathKind},
    resolver::{resolver_for_scope, Resolver, TypeNs, ValueNs},
    AsMacroCall, DefWithBodyId, FieldId, FunctionId, LocalFieldId, VariantId,
};
use hir_expand::{hygiene::Hygiene, name::AsName, HirFileId, InFile};
use hir_ty::{
    diagnostics::{record_literal_missing_fields, record_pattern_missing_fields},
    InferenceResult, Interner, Substitution, TyLoweringContext,
};
use syntax::{
    ast::{self, AstNode},
    SyntaxNode, TextRange, TextSize,
};

use crate::{
    db::HirDatabase, semantics::PathResolution, Adt, BuiltinType, Const, Field, Function, Local,
    MacroDef, ModuleDef, Static, Struct, Trait, Type, TypeAlias, TypeParam, Variant,
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
    scopes: Option<Arc<ExprScopes>>,
}

impl SourceAnalyzer {
    pub(crate) fn new_for_body(
        db: &dyn HirDatabase,
        def: DefWithBodyId,
        node: InFile<&SyntaxNode>,
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
            scopes: Some(scopes),
            file_id: node.file_id,
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
            scopes: None,
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

    pub(crate) fn type_of_expr(&self, db: &dyn HirDatabase, expr: &ast::Expr) -> Option<Type> {
        let expr_id = self.expr_id(db, expr)?;
        let ty = self.infer.as_ref()?[expr_id].clone();
        Type::new_with_resolver(db, &self.resolver, ty)
    }

    pub(crate) fn type_of_pat(&self, db: &dyn HirDatabase, pat: &ast::Pat) -> Option<Type> {
        let pat_id = self.pat_id(pat)?;
        let ty = self.infer.as_ref()?[pat_id].clone();
        Type::new_with_resolver(db, &self.resolver, ty)
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
    ) -> Option<FunctionId> {
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
    ) -> Option<(Field, Option<Local>)> {
        let expr_id =
            self.body_source_map.as_ref()?.node_field(InFile::new(self.file_id, field))?;

        let local = if field.name_ref().is_some() {
            None
        } else {
            let local_name = field.field_name()?.as_name();
            let path = ModPath::from_segments(PathKind::Plain, once(local_name));
            match self.resolver.resolve_path_in_value_ns_fully(db.upcast(), &path) {
                Some(ValueNs::LocalBinding(pat_id)) => {
                    Some(Local { pat_id, parent: self.resolver.body_owner()? })
                }
                _ => None,
            }
        };
        let struct_field = self.infer.as_ref()?.record_field_resolution(expr_id)?;
        Some((struct_field.into(), local))
    }

    pub(crate) fn resolve_record_pat_field(
        &self,
        _db: &dyn HirDatabase,
        field: &ast::RecordPatField,
    ) -> Option<Field> {
        let pat_id = self.pat_id(&field.pat()?)?;
        let struct_field = self.infer.as_ref()?.record_pat_field_resolution(pat_id)?;
        Some(struct_field.into())
    }

    pub(crate) fn resolve_macro_call(
        &self,
        db: &dyn HirDatabase,
        macro_call: InFile<&ast::MacroCall>,
    ) -> Option<MacroDef> {
        let hygiene = Hygiene::new(db.upcast(), macro_call.file_id);
        let path = macro_call.value.path().and_then(|ast| Path::from_src(ast, &hygiene))?;
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
        let res = resolve_hir_path(db, &self.resolver, &path)?;
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
        let parent = || path.syntax().parent();
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
        }

        if let Some(path_pat) = parent().and_then(ast::PathPat::cast) {
            let pat_id = self.pat_id(&path_pat.into())?;
            if let Some(assoc) = self.infer.as_ref()?.assoc_resolutions_for_pat(pat_id) {
                return Some(PathResolution::AssocItem(assoc.into()));
            }
            if let Some(VariantId::EnumVariantId(variant)) =
                self.infer.as_ref()?.variant_resolution_for_pat(pat_id)
            {
                return Some(PathResolution::Def(ModuleDef::Variant(variant.into())));
            }
        }

        if let Some(rec_lit) = parent().and_then(ast::RecordExpr::cast) {
            let expr_id = self.expr_id(db, &rec_lit.into())?;
            if let Some(VariantId::EnumVariantId(variant)) =
                self.infer.as_ref()?.variant_resolution_for_expr(expr_id)
            {
                return Some(PathResolution::Def(ModuleDef::Variant(variant.into())));
            }
        }

        if let Some(pat) = parent()
            .and_then(ast::RecordPat::cast)
            .map(ast::Pat::from)
            .or_else(|| parent().and_then(ast::TupleStructPat::cast).map(ast::Pat::from))
        {
            let pat_id = self.pat_id(&pat)?;
            if let Some(VariantId::EnumVariantId(variant)) =
                self.infer.as_ref()?.variant_resolution_for_pat(pat_id)
            {
                return Some(PathResolution::Def(ModuleDef::Variant(variant.into())));
            }
        }

        // This must be a normal source file rather than macro file.
        let hir_path = Path::from_src(path.clone(), &Hygiene::new(db.upcast(), self.file_id))?;

        // Case where path is a qualifier of another path, e.g. foo::bar::Baz where we
        // trying to resolve foo::bar.
        if let Some(outer_path) = parent().and_then(ast::Path::cast) {
            if let Some(qualifier) = outer_path.qualifier() {
                if path == &qualifier {
                    return resolve_hir_path_qualifier(db, &self.resolver, &hir_path);
                }
            }
        }

        resolve_hir_path_(db, &self.resolver, &hir_path, prefer_value_ns)
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
        let substs = infer.type_of_expr[expr_id].substs()?;

        let (variant, missing_fields, _exhaustive) =
            record_literal_missing_fields(db, infer, expr_id, &body[expr_id])?;
        let res = self.missing_fields(db, krate, &substs, variant, missing_fields);
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
        let substs = infer.type_of_pat[pat_id].substs()?;

        let (variant, missing_fields, _exhaustive) =
            record_pattern_missing_fields(db, infer, pat_id, &body[pat_id])?;
        let res = self.missing_fields(db, krate, &substs, variant, missing_fields);
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
                let ty = field_types[local_id].clone().substitute(&Interner, substs);
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
            self.resolver.resolve_path_as_macro(db.upcast(), &path)
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
            let source = source_map.expr_syntax(*id).ok()?;
            // FIXME: correctly handle macro expansion
            if source.file_id != offset.file_id {
                return None;
            }
            let root = source.file_syntax(db.upcast());
            let node = source.value.to_node(&root);
            Some((node.syntax().text_range(), scope))
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
        let res = match ty {
            TypeNs::SelfType(it) => PathResolution::SelfType(it.into()),
            TypeNs::GenericParam(id) => PathResolution::TypeParam(TypeParam { id }),
            TypeNs::AdtSelfType(it) | TypeNs::AdtId(it) => {
                PathResolution::Def(Adt::from(it).into())
            }
            TypeNs::EnumVariantId(it) => PathResolution::Def(Variant::from(it).into()),
            TypeNs::TypeAliasId(it) => PathResolution::Def(TypeAlias::from(it).into()),
            TypeNs::BuiltinType(it) => PathResolution::Def(BuiltinType::from(it).into()),
            TypeNs::TraitId(it) => PathResolution::Def(Trait::from(it).into()),
        };
        match unresolved {
            Some(unresolved) => res
                .assoc_type_shorthand_candidates(db, |name, alias| {
                    (name == unresolved.name).then(|| alias)
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
                ValueNs::GenericParam(it) => PathResolution::ConstParam(it.into()),
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
            .map(|def| PathResolution::Macro(def.into()))
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
        TypeNs::GenericParam(id) => PathResolution::TypeParam(TypeParam { id }),
        TypeNs::AdtSelfType(it) | TypeNs::AdtId(it) => PathResolution::Def(Adt::from(it).into()),
        TypeNs::EnumVariantId(it) => PathResolution::Def(Variant::from(it).into()),
        TypeNs::TypeAliasId(it) => PathResolution::Def(TypeAlias::from(it).into()),
        TypeNs::BuiltinType(it) => PathResolution::Def(BuiltinType::from(it).into()),
        TypeNs::TraitId(it) => PathResolution::Def(Trait::from(it).into()),
    })
}
