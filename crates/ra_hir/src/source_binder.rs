//! Lookup hir elements using positions in the source code. This is a lossy
//! transformation: in general, a single source might correspond to several
//! modules, functions, etc, due to macros, cfgs and `#[path=]` attributes on
//! modules.
//!
//! So, this modules should not be used during hir construction, it exists
//! purely for "IDE needs".
use std::sync::Arc;

use hir_def::path::known;
use hir_expand::name::AsName;
use ra_db::FileId;
use ra_syntax::{
    ast::{self, AstNode},
    match_ast, AstPtr,
    SyntaxKind::*,
    SyntaxNode, SyntaxNodePtr, TextRange, TextUnit,
};
use rustc_hash::FxHashSet;

use crate::{
    db::HirDatabase,
    expr::{
        self,
        scope::{ExprScopes, ScopeId},
        BodySourceMap,
    },
    ids::LocationCtx,
    resolve::{ScopeDef, TypeNs, ValueNs},
    ty::method_resolution::{self, implements_trait},
    AssocItem, Const, DefWithBody, Either, Enum, FromSource, Function, HasBody, HirFileId,
    MacroDef, Module, Name, Path, Resolver, Static, Struct, Ty,
};

fn try_get_resolver_for_node(
    db: &impl HirDatabase,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<Resolver> {
    match_ast! {
        match node {
            ast::Module(it) => {
                let src = crate::Source { file_id: file_id.into(), ast: it };
                Some(crate::Module::from_declaration(db, src)?.resolver(db))
            },
             ast::SourceFile(it) => {
                let src =
                    crate::Source { file_id: file_id.into(), ast: crate::ModuleSource::SourceFile(it) };
                Some(crate::Module::from_definition(db, src)?.resolver(db))
            },
            ast::StructDef(it) => {
                let src = crate::Source { file_id: file_id.into(), ast: it };
                Some(Struct::from_source(db, src)?.resolver(db))
            },
            ast::EnumDef(it) => {
                let src = crate::Source { file_id: file_id.into(), ast: it };
                Some(Enum::from_source(db, src)?.resolver(db))
            },
            _ => {
                if node.kind() == FN_DEF || node.kind() == CONST_DEF || node.kind() == STATIC_DEF {
                    Some(def_with_body_from_child_node(db, file_id, node)?.resolver(db))
                } else {
                    // FIXME add missing cases
                    None
                }
            },
        }
    }
}

fn def_with_body_from_child_node(
    db: &impl HirDatabase,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<DefWithBody> {
    let src = crate::ModuleSource::from_child_node(db, file_id, node);
    let module = Module::from_definition(db, crate::Source { file_id: file_id.into(), ast: src })?;
    let ctx = LocationCtx::new(db, module.id, file_id.into());

    node.ancestors().find_map(|node| {
        match_ast! {
            match node {
                ast::FnDef(def)  => { Some(Function {id: ctx.to_def(&def) }.into()) },
                ast::ConstDef(def) => { Some(Const { id: ctx.to_def(&def) }.into()) },
                ast::StaticDef(def) => { Some(Static { id: ctx.to_def(&def) }.into()) },
                _ => { None },
            }
        }
    })
}

/// `SourceAnalyzer` is a convenience wrapper which exposes HIR API in terms of
/// original source files. It should not be used inside the HIR itself.
#[derive(Debug)]
pub struct SourceAnalyzer {
    resolver: Resolver,
    body_source_map: Option<Arc<BodySourceMap>>,
    infer: Option<Arc<crate::ty::InferenceResult>>,
    scopes: Option<Arc<crate::expr::ExprScopes>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathResolution {
    /// An item
    Def(crate::ModuleDef),
    /// A local binding (only value namespace)
    LocalBinding(Either<AstPtr<ast::BindPat>, AstPtr<ast::SelfParam>>),
    /// A generic parameter
    GenericParam(u32),
    SelfType(crate::ImplBlock),
    Macro(MacroDef),
    AssocItem(crate::AssocItem),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeEntryWithSyntax {
    pub(crate) name: Name,
    pub(crate) ptr: Either<AstPtr<ast::Pat>, AstPtr<ast::SelfParam>>,
}

impl ScopeEntryWithSyntax {
    pub fn name(&self) -> &Name {
        &self.name
    }

    pub fn ptr(&self) -> Either<AstPtr<ast::Pat>, AstPtr<ast::SelfParam>> {
        self.ptr
    }
}

#[derive(Debug)]
pub struct ReferenceDescriptor {
    pub range: TextRange,
    pub name: String,
}

impl SourceAnalyzer {
    pub fn new(
        db: &impl HirDatabase,
        file_id: FileId,
        node: &SyntaxNode,
        offset: Option<TextUnit>,
    ) -> SourceAnalyzer {
        let def_with_body = def_with_body_from_child_node(db, file_id, node);
        if let Some(def) = def_with_body {
            let source_map = def.body_source_map(db);
            let scopes = db.expr_scopes(def);
            let scope = match offset {
                None => scope_for(&scopes, &source_map, &node),
                Some(offset) => scope_for_offset(&scopes, &source_map, file_id.into(), offset),
            };
            let resolver = expr::resolver_for_scope(def.body(db), db, scope);
            SourceAnalyzer {
                resolver,
                body_source_map: Some(source_map),
                infer: Some(def.infer(db)),
                scopes: Some(scopes),
            }
        } else {
            SourceAnalyzer {
                resolver: node
                    .ancestors()
                    .find_map(|node| try_get_resolver_for_node(db, file_id, &node))
                    .unwrap_or_default(),
                body_source_map: None,
                infer: None,
                scopes: None,
            }
        }
    }

    pub fn type_of(&self, _db: &impl HirDatabase, expr: &ast::Expr) -> Option<crate::Ty> {
        let expr_id = self.body_source_map.as_ref()?.node_expr(expr)?;
        Some(self.infer.as_ref()?[expr_id].clone())
    }

    pub fn type_of_pat(&self, _db: &impl HirDatabase, pat: &ast::Pat) -> Option<crate::Ty> {
        let pat_id = self.body_source_map.as_ref()?.node_pat(pat)?;
        Some(self.infer.as_ref()?[pat_id].clone())
    }

    pub fn type_of_pat_by_id(
        &self,
        _db: &impl HirDatabase,
        pat_id: expr::PatId,
    ) -> Option<crate::Ty> {
        Some(self.infer.as_ref()?[pat_id].clone())
    }

    pub fn resolve_method_call(&self, call: &ast::MethodCallExpr) -> Option<Function> {
        let expr_id = self.body_source_map.as_ref()?.node_expr(&call.clone().into())?;
        self.infer.as_ref()?.method_resolution(expr_id)
    }

    pub fn resolve_field(&self, field: &ast::FieldExpr) -> Option<crate::StructField> {
        let expr_id = self.body_source_map.as_ref()?.node_expr(&field.clone().into())?;
        self.infer.as_ref()?.field_resolution(expr_id)
    }

    pub fn resolve_record_literal(&self, record_lit: &ast::RecordLit) -> Option<crate::VariantDef> {
        let expr_id = self.body_source_map.as_ref()?.node_expr(&record_lit.clone().into())?;
        self.infer.as_ref()?.variant_resolution_for_expr(expr_id)
    }

    pub fn resolve_record_pattern(&self, record_pat: &ast::RecordPat) -> Option<crate::VariantDef> {
        let pat_id = self.body_source_map.as_ref()?.node_pat(&record_pat.clone().into())?;
        self.infer.as_ref()?.variant_resolution_for_pat(pat_id)
    }

    pub fn resolve_macro_call(
        &self,
        db: &impl HirDatabase,
        macro_call: &ast::MacroCall,
    ) -> Option<MacroDef> {
        // This must be a normal source file rather than macro file.
        let path = macro_call.path().and_then(Path::from_ast)?;
        self.resolver.resolve_path_as_macro(db, &path)
    }

    pub fn resolve_hir_path(
        &self,
        db: &impl HirDatabase,
        path: &crate::Path,
    ) -> Option<PathResolution> {
        let types = self.resolver.resolve_path_in_type_ns_fully(db, &path).map(|ty| match ty {
            TypeNs::SelfType(it) => PathResolution::SelfType(it),
            TypeNs::GenericParam(it) => PathResolution::GenericParam(it),
            TypeNs::AdtSelfType(it) | TypeNs::Adt(it) => PathResolution::Def(it.into()),
            TypeNs::EnumVariant(it) => PathResolution::Def(it.into()),
            TypeNs::TypeAlias(it) => PathResolution::Def(it.into()),
            TypeNs::BuiltinType(it) => PathResolution::Def(it.into()),
            TypeNs::Trait(it) => PathResolution::Def(it.into()),
        });
        let values = self.resolver.resolve_path_in_value_ns_fully(db, &path).and_then(|val| {
            let res = match val {
                ValueNs::LocalBinding(it) => {
                    // We get a `PatId` from resolver, but it actually can only
                    // point at `BindPat`, and not at the arbitrary pattern.
                    let pat_ptr = self
                        .body_source_map
                        .as_ref()?
                        .pat_syntax(it)?
                        .ast // FIXME: ignoring file_id here is definitelly wrong
                        .map_a(|ptr| ptr.cast::<ast::BindPat>().unwrap());
                    PathResolution::LocalBinding(pat_ptr)
                }
                ValueNs::Function(it) => PathResolution::Def(it.into()),
                ValueNs::Const(it) => PathResolution::Def(it.into()),
                ValueNs::Static(it) => PathResolution::Def(it.into()),
                ValueNs::Struct(it) => PathResolution::Def(it.into()),
                ValueNs::EnumVariant(it) => PathResolution::Def(it.into()),
            };
            Some(res)
        });

        let items = self
            .resolver
            .resolve_module_path(db, &path)
            .take_types()
            .map(|it| PathResolution::Def(it.into()));
        types.or(values).or(items).or_else(|| {
            self.resolver.resolve_path_as_macro(db, &path).map(|def| PathResolution::Macro(def))
        })
    }

    pub fn resolve_path(&self, db: &impl HirDatabase, path: &ast::Path) -> Option<PathResolution> {
        if let Some(path_expr) = path.syntax().parent().and_then(ast::PathExpr::cast) {
            let expr_id = self.body_source_map.as_ref()?.node_expr(&path_expr.into())?;
            if let Some(assoc) = self.infer.as_ref()?.assoc_resolutions_for_expr(expr_id) {
                return Some(PathResolution::AssocItem(assoc));
            }
        }
        if let Some(path_pat) = path.syntax().parent().and_then(ast::PathPat::cast) {
            let pat_id = self.body_source_map.as_ref()?.node_pat(&path_pat.into())?;
            if let Some(assoc) = self.infer.as_ref()?.assoc_resolutions_for_pat(pat_id) {
                return Some(PathResolution::AssocItem(assoc));
            }
        }
        // This must be a normal source file rather than macro file.
        let hir_path = crate::Path::from_ast(path.clone())?;
        self.resolve_hir_path(db, &hir_path)
    }

    pub fn resolve_local_name(&self, name_ref: &ast::NameRef) -> Option<ScopeEntryWithSyntax> {
        let mut shadowed = FxHashSet::default();
        let name = name_ref.as_name();
        let source_map = self.body_source_map.as_ref()?;
        let scopes = self.scopes.as_ref()?;
        let scope = scope_for(scopes, source_map, name_ref.syntax());
        let ret = scopes
            .scope_chain(scope)
            .flat_map(|scope| scopes.entries(scope).iter())
            .filter(|entry| shadowed.insert(entry.name()))
            .filter(|entry| entry.name() == &name)
            .nth(0);
        ret.and_then(|entry| {
            Some(ScopeEntryWithSyntax {
                name: entry.name().clone(),
                ptr: source_map.pat_syntax(entry.pat())?.ast,
            })
        })
    }

    pub fn process_all_names(&self, db: &impl HirDatabase, f: &mut dyn FnMut(Name, ScopeDef)) {
        self.resolver.process_all_names(db, f)
    }

    pub fn find_all_refs(&self, pat: &ast::BindPat) -> Vec<ReferenceDescriptor> {
        // FIXME: at least, this should work with any DefWithBody, but ideally
        // this should be hir-based altogether
        let fn_def = pat.syntax().ancestors().find_map(ast::FnDef::cast).unwrap();
        let ptr = Either::A(AstPtr::new(&ast::Pat::from(pat.clone())));
        fn_def
            .syntax()
            .descendants()
            .filter_map(ast::NameRef::cast)
            .filter(|name_ref| match self.resolve_local_name(&name_ref) {
                None => false,
                Some(entry) => entry.ptr() == ptr,
            })
            .map(|name_ref| ReferenceDescriptor {
                name: name_ref.text().to_string(),
                range: name_ref.syntax().text_range(),
            })
            .collect()
    }

    pub fn iterate_method_candidates<T>(
        &self,
        db: &impl HirDatabase,
        ty: Ty,
        name: Option<&Name>,
        mut callback: impl FnMut(&Ty, Function) -> Option<T>,
    ) -> Option<T> {
        // There should be no inference vars in types passed here
        // FIXME check that?
        // FIXME replace Unknown by bound vars here
        let canonical = crate::ty::Canonical { value: ty, num_vars: 0 };
        method_resolution::iterate_method_candidates(
            &canonical,
            db,
            &self.resolver,
            name,
            method_resolution::LookupMode::MethodCall,
            |ty, it| match it {
                AssocItem::Function(f) => callback(ty, f),
                _ => None,
            },
        )
    }

    pub fn iterate_path_candidates<T>(
        &self,
        db: &impl HirDatabase,
        ty: Ty,
        name: Option<&Name>,
        callback: impl FnMut(&Ty, AssocItem) -> Option<T>,
    ) -> Option<T> {
        // There should be no inference vars in types passed here
        // FIXME check that?
        // FIXME replace Unknown by bound vars here
        let canonical = crate::ty::Canonical { value: ty, num_vars: 0 };
        method_resolution::iterate_method_candidates(
            &canonical,
            db,
            &self.resolver,
            name,
            method_resolution::LookupMode::Path,
            callback,
        )
    }

    pub fn autoderef<'a>(
        &'a self,
        db: &'a impl HirDatabase,
        ty: Ty,
    ) -> impl Iterator<Item = Ty> + 'a {
        // There should be no inference vars in types passed here
        // FIXME check that?
        let canonical = crate::ty::Canonical { value: ty, num_vars: 0 };
        crate::ty::autoderef(db, &self.resolver, canonical).map(|canonical| canonical.value)
    }

    /// Checks that particular type `ty` implements `std::future::Future`.
    /// This function is used in `.await` syntax completion.
    pub fn impls_future(&self, db: &impl HirDatabase, ty: Ty) -> bool {
        let std_future_path = known::std_future_future();

        let std_future_trait = match self.resolver.resolve_known_trait(db, &std_future_path) {
            Some(it) => it,
            _ => return false,
        };

        let krate = match self.resolver.krate() {
            Some(krate) => krate,
            _ => return false,
        };

        let canonical_ty = crate::ty::Canonical { value: ty, num_vars: 0 };
        implements_trait(&canonical_ty, db, &self.resolver, krate, std_future_trait)
    }

    #[cfg(test)]
    pub(crate) fn body_source_map(&self) -> Arc<BodySourceMap> {
        self.body_source_map.clone().unwrap()
    }

    #[cfg(test)]
    pub(crate) fn inference_result(&self) -> Arc<crate::ty::InferenceResult> {
        self.infer.clone().unwrap()
    }

    #[cfg(test)]
    pub(crate) fn scopes(&self) -> Arc<ExprScopes> {
        self.scopes.clone().unwrap()
    }
}

fn scope_for(
    scopes: &ExprScopes,
    source_map: &BodySourceMap,
    node: &SyntaxNode,
) -> Option<ScopeId> {
    node.ancestors()
        .filter_map(ast::Expr::cast)
        .filter_map(|it| source_map.node_expr(&it))
        .find_map(|it| scopes.scope_for(it))
}

fn scope_for_offset(
    scopes: &ExprScopes,
    source_map: &BodySourceMap,
    file_id: HirFileId,
    offset: TextUnit,
) -> Option<ScopeId> {
    scopes
        .scope_by_expr()
        .iter()
        .filter_map(|(id, scope)| {
            let source = source_map.expr_syntax(*id)?;
            // FIXME: correctly handle macro expansion
            if source.file_id != file_id {
                return None;
            }
            let syntax_node_ptr =
                source.ast.either(|it| it.syntax_node_ptr(), |it| it.syntax_node_ptr());
            Some((syntax_node_ptr, scope))
        })
        // find containing scope
        .min_by_key(|(ptr, _scope)| {
            (!(ptr.range().start() <= offset && offset <= ptr.range().end()), ptr.range().len())
        })
        .map(|(ptr, scope)| adjust(scopes, source_map, ptr, file_id, offset).unwrap_or(*scope))
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
            let source = source_map.expr_syntax(*id)?;
            // FIXME: correctly handle macro expansion
            if source.file_id != file_id {
                return None;
            }
            let syntax_node_ptr =
                source.ast.either(|it| it.syntax_node_ptr(), |it| it.syntax_node_ptr());
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
