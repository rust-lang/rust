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
    resolver::{resolver_for_scope, Resolver, TypeNs, ValueNs},
    AsMacroCall, DefWithBodyId,
};
use hir_expand::{hygiene::Hygiene, name::AsName, HirFileId, InFile};
use hir_ty::InferenceResult;
use ra_syntax::{
    ast::{self, AstNode},
    SyntaxNode, SyntaxNodePtr, TextUnit,
};

use crate::{
    db::HirDatabase, semantics::PathResolution, Adt, Const, EnumVariant, Function, Local, MacroDef,
    ModPath, ModuleDef, Path, PathKind, Static, Struct, Trait, Type, TypeAlias, TypeParam,
};

/// `SourceAnalyzer` is a convenience wrapper which exposes HIR API in terms of
/// original source files. It should not be used inside the HIR itself.
#[derive(Debug)]
pub(crate) struct SourceAnalyzer {
    file_id: HirFileId,
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
        offset: Option<TextUnit>,
    ) -> SourceAnalyzer {
        let (body, source_map) = db.body_with_source_map(def);
        let scopes = db.expr_scopes(def);
        let scope = match offset {
            None => scope_for(&scopes, &source_map, node),
            Some(offset) => scope_for_offset(&scopes, &source_map, node.with_value(offset)),
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

    fn expr_id(&self, expr: &ast::Expr) -> Option<ExprId> {
        let src = InFile { file_id: self.file_id, value: expr };
        self.body_source_map.as_ref()?.node_expr(src)
    }

    fn pat_id(&self, pat: &ast::Pat) -> Option<PatId> {
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

    pub(crate) fn type_of(&self, db: &dyn HirDatabase, expr: &ast::Expr) -> Option<Type> {
        let expr_id = match expr {
            ast::Expr::MacroCall(call) => {
                let expr = self.expand_expr(db, InFile::new(self.file_id, call.clone()))?;
                self.body_source_map.as_ref()?.node_expr(expr.as_ref())
            }
            _ => self.expr_id(expr),
        }?;

        let ty = self.infer.as_ref()?[expr_id].clone();
        Type::new_with_resolver(db, &self.resolver, ty)
    }

    pub(crate) fn type_of_pat(&self, db: &dyn HirDatabase, pat: &ast::Pat) -> Option<Type> {
        let pat_id = self.pat_id(pat)?;
        let ty = self.infer.as_ref()?[pat_id].clone();
        Type::new_with_resolver(db, &self.resolver, ty)
    }

    pub(crate) fn resolve_method_call(&self, call: &ast::MethodCallExpr) -> Option<Function> {
        let expr_id = self.expr_id(&call.clone().into())?;
        self.infer.as_ref()?.method_resolution(expr_id).map(Function::from)
    }

    pub(crate) fn resolve_field(&self, field: &ast::FieldExpr) -> Option<crate::StructField> {
        let expr_id = self.expr_id(&field.clone().into())?;
        self.infer.as_ref()?.field_resolution(expr_id).map(|it| it.into())
    }

    pub(crate) fn resolve_record_field(
        &self,
        db: &dyn HirDatabase,
        field: &ast::RecordField,
    ) -> Option<(crate::StructField, Option<Local>)> {
        let (expr_id, local) = match field.expr() {
            Some(it) => (self.expr_id(&it)?, None),
            None => {
                let src = InFile { file_id: self.file_id, value: field };
                let expr_id = self.body_source_map.as_ref()?.field_init_shorthand_expr(src)?;
                let local_name = field.name_ref()?.as_name();
                let path = ModPath::from_segments(PathKind::Plain, once(local_name));
                let local = match self.resolver.resolve_path_in_value_ns_fully(db.upcast(), &path) {
                    Some(ValueNs::LocalBinding(pat_id)) => {
                        Some(Local { pat_id, parent: self.resolver.body_owner()? })
                    }
                    _ => None,
                };
                (expr_id, local)
            }
        };
        let struct_field = self.infer.as_ref()?.record_field_resolution(expr_id)?;
        Some((struct_field.into(), local))
    }

    pub(crate) fn resolve_record_literal(
        &self,
        record_lit: &ast::RecordLit,
    ) -> Option<crate::VariantDef> {
        let expr_id = self.expr_id(&record_lit.clone().into())?;
        self.infer.as_ref()?.variant_resolution_for_expr(expr_id).map(|it| it.into())
    }

    pub(crate) fn resolve_record_pattern(
        &self,
        record_pat: &ast::RecordPat,
    ) -> Option<crate::VariantDef> {
        let pat_id = self.pat_id(&record_pat.clone().into())?;
        self.infer.as_ref()?.variant_resolution_for_pat(pat_id).map(|it| it.into())
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
        pat: &ast::BindPat,
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
        if let Some(path_expr) = path.syntax().parent().and_then(ast::PathExpr::cast) {
            let expr_id = self.expr_id(&path_expr.into())?;
            if let Some(assoc) = self.infer.as_ref()?.assoc_resolutions_for_expr(expr_id) {
                return Some(PathResolution::AssocItem(assoc.into()));
            }
        }
        if let Some(path_pat) = path.syntax().parent().and_then(ast::PathPat::cast) {
            let pat_id = self.pat_id(&path_pat.into())?;
            if let Some(assoc) = self.infer.as_ref()?.assoc_resolutions_for_pat(pat_id) {
                return Some(PathResolution::AssocItem(assoc.into()));
            }
        }
        // This must be a normal source file rather than macro file.
        let hir_path = crate::Path::from_ast(path.clone())?;
        resolve_hir_path(db, &self.resolver, &hir_path)
    }

    pub(crate) fn expand(
        &self,
        db: &dyn HirDatabase,
        macro_call: InFile<&ast::MacroCall>,
    ) -> Option<HirFileId> {
        let macro_call_id = macro_call.as_call_id(db.upcast(), |path| {
            self.resolver.resolve_path_as_macro(db.upcast(), &path)
        })?;
        Some(macro_call_id.as_file())
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
    scopes: &ExprScopes,
    source_map: &BodySourceMap,
    offset: InFile<TextUnit>,
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
            let syntax_node_ptr =
                source.value.either(|it| it.syntax_node_ptr(), |it| it.syntax_node_ptr());
            Some((syntax_node_ptr, scope))
        })
        // find containing scope
        .min_by_key(|(ptr, _scope)| {
            (
                !(ptr.range().start() <= offset.value && offset.value <= ptr.range().end()),
                ptr.range().len(),
            )
        })
        .map(|(ptr, scope)| {
            adjust(scopes, source_map, ptr, offset.file_id, offset.value).unwrap_or(*scope)
        })
}

pub(crate) fn resolve_hir_path(
    db: &dyn HirDatabase,
    resolver: &Resolver,
    path: &crate::Path,
) -> Option<PathResolution> {
    let types =
        resolver.resolve_path_in_type_ns_fully(db.upcast(), path.mod_path()).map(|ty| match ty {
            TypeNs::SelfType(it) => PathResolution::SelfType(it.into()),
            TypeNs::GenericParam(id) => PathResolution::TypeParam(TypeParam { id }),
            TypeNs::AdtSelfType(it) | TypeNs::AdtId(it) => {
                PathResolution::Def(Adt::from(it).into())
            }
            TypeNs::EnumVariantId(it) => PathResolution::Def(EnumVariant::from(it).into()),
            TypeNs::TypeAliasId(it) => PathResolution::Def(TypeAlias::from(it).into()),
            TypeNs::BuiltinType(it) => PathResolution::Def(it.into()),
            TypeNs::TraitId(it) => PathResolution::Def(Trait::from(it).into()),
        });
    let body_owner = resolver.body_owner();
    let values =
        resolver.resolve_path_in_value_ns_fully(db.upcast(), path.mod_path()).and_then(|val| {
            let res = match val {
                ValueNs::LocalBinding(pat_id) => {
                    let var = Local { parent: body_owner?.into(), pat_id };
                    PathResolution::Local(var)
                }
                ValueNs::FunctionId(it) => PathResolution::Def(Function::from(it).into()),
                ValueNs::ConstId(it) => PathResolution::Def(Const::from(it).into()),
                ValueNs::StaticId(it) => PathResolution::Def(Static::from(it).into()),
                ValueNs::StructId(it) => PathResolution::Def(Struct::from(it).into()),
                ValueNs::EnumVariantId(it) => PathResolution::Def(EnumVariant::from(it).into()),
            };
            Some(res)
        });

    let items = resolver
        .resolve_module_path_in_items(db.upcast(), path.mod_path())
        .take_types()
        .map(|it| PathResolution::Def(it.into()));
    types.or(values).or(items).or_else(|| {
        resolver
            .resolve_path_as_macro(db.upcast(), path.mod_path())
            .map(|def| PathResolution::Macro(def.into()))
    })
}

// XXX: during completion, cursor might be outside of any particular
// expression. Try to figure out the correct scope...
fn adjust(
    scopes: &ExprScopes,
    source_map: &BodySourceMap,
    ptr: SyntaxNodePtr,
    file_id: HirFileId,
    offset: TextUnit,
) -> Option<ScopeId> {
    let r = ptr.range();
    let child_scopes = scopes
        .scope_by_expr()
        .iter()
        .filter_map(|(id, scope)| {
            let source = source_map.expr_syntax(*id).ok()?;
            // FIXME: correctly handle macro expansion
            if source.file_id != file_id {
                return None;
            }
            let syntax_node_ptr =
                source.value.either(|it| it.syntax_node_ptr(), |it| it.syntax_node_ptr());
            Some((syntax_node_ptr, scope))
        })
        .map(|(ptr, scope)| (ptr.range(), scope))
        .filter(|(range, _)| range.start() <= offset && range.is_subrange(&r) && *range != r);

    child_scopes
        .max_by(|(r1, _), (r2, _)| {
            if r2.is_subrange(&r1) {
                std::cmp::Ordering::Greater
            } else if r1.is_subrange(&r2) {
                std::cmp::Ordering::Less
            } else {
                r1.start().cmp(&r2.start())
            }
        })
        .map(|(_ptr, scope)| *scope)
}
