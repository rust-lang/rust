/// Lookup hir elements using positions in the source code. This is a lossy
/// transformation: in general, a single source might correspond to several
/// modules, functions, etc, due to macros, cfgs and `#[path=]` attributes on
/// modules.
///
/// So, this modules should not be used during hir construction, it exists
/// purely for "IDE needs".
use std::sync::Arc;

use ra_db::{FileId, FilePosition};
use ra_syntax::{
    algo::find_node_at_offset,
    ast::{self, AstNode, NameOwner},
    AstPtr,
    SyntaxKind::*,
    SyntaxNode, SyntaxNodePtr, TextRange, TextUnit,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    expr,
    expr::{
        scope::{ExprScopes, ScopeId},
        BodySourceMap,
    },
    ids::LocationCtx,
    AsName, AstId, Const, Crate, DefWithBody, Either, Enum, Function, HirDatabase, HirFileId,
    MacroDef, Module, Name, Path, PerNs, Resolver, Static, Struct, Trait, Ty,
};

/// Locates the module by `FileId`. Picks topmost module in the file.
pub fn module_from_file_id(db: &impl HirDatabase, file_id: FileId) -> Option<Module> {
    module_from_source(db, file_id.into(), None)
}

/// Locates the child module by `mod child;` declaration.
pub fn module_from_declaration(
    db: &impl HirDatabase,
    file_id: FileId,
    decl: ast::Module,
) -> Option<Module> {
    let parent_module = module_from_file_id(db, file_id);
    let child_name = decl.name();
    match (parent_module, child_name) {
        (Some(parent_module), Some(child_name)) => parent_module.child(db, &child_name.as_name()),
        _ => None,
    }
}

/// Locates the module by position in the source code.
pub fn module_from_position(db: &impl HirDatabase, position: FilePosition) -> Option<Module> {
    let parse = db.parse(position.file_id);
    match &find_node_at_offset::<ast::Module>(parse.tree().syntax(), position.offset) {
        Some(m) if !m.has_semi() => module_from_inline(db, position.file_id, m.clone()),
        _ => module_from_file_id(db, position.file_id),
    }
}

fn module_from_inline(
    db: &impl HirDatabase,
    file_id: FileId,
    module: ast::Module,
) -> Option<Module> {
    assert!(!module.has_semi());
    let file_id = file_id.into();
    let ast_id_map = db.ast_id_map(file_id);
    let item_id = ast_id_map.ast_id(&module).with_file_id(file_id);
    module_from_source(db, file_id, Some(item_id))
}

/// Locates the module by child syntax element within the module
pub fn module_from_child_node(
    db: &impl HirDatabase,
    file_id: FileId,
    child: &SyntaxNode,
) -> Option<Module> {
    if let Some(m) = child.ancestors().filter_map(ast::Module::cast).find(|it| !it.has_semi()) {
        module_from_inline(db, file_id, m)
    } else {
        module_from_file_id(db, file_id)
    }
}

fn module_from_source(
    db: &impl HirDatabase,
    file_id: HirFileId,
    decl_id: Option<AstId<ast::Module>>,
) -> Option<Module> {
    let source_root_id = db.file_source_root(file_id.as_original_file());
    db.source_root_crates(source_root_id).iter().map(|&crate_id| Crate { crate_id }).find_map(
        |krate| {
            let def_map = db.crate_def_map(krate);
            let module_id = def_map.find_module_by_source(file_id, decl_id)?;
            Some(Module { krate, module_id })
        },
    )
}

pub fn struct_from_module(
    db: &impl HirDatabase,
    module: Module,
    struct_def: &ast::StructDef,
) -> Struct {
    let file_id = module.definition_source(db).file_id;
    let ctx = LocationCtx::new(db, module, file_id);
    Struct { id: ctx.to_def(struct_def) }
}

pub fn enum_from_module(db: &impl HirDatabase, module: Module, enum_def: &ast::EnumDef) -> Enum {
    let file_id = module.definition_source(db).file_id;
    let ctx = LocationCtx::new(db, module, file_id);
    Enum { id: ctx.to_def(enum_def) }
}

pub fn trait_from_module(
    db: &impl HirDatabase,
    module: Module,
    trait_def: &ast::TraitDef,
) -> Trait {
    let file_id = module.definition_source(db).file_id;
    let ctx = LocationCtx::new(db, module, file_id);
    Trait { id: ctx.to_def(trait_def) }
}

fn try_get_resolver_for_node(
    db: &impl HirDatabase,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<Resolver> {
    if let Some(module) = ast::Module::cast(node.clone()) {
        Some(module_from_declaration(db, file_id, module)?.resolver(db))
    } else if let Some(_) = ast::SourceFile::cast(node.clone()) {
        Some(module_from_source(db, file_id.into(), None)?.resolver(db))
    } else if let Some(s) = ast::StructDef::cast(node.clone()) {
        let module = module_from_child_node(db, file_id, s.syntax())?;
        Some(struct_from_module(db, module, &s).resolver(db))
    } else if let Some(e) = ast::EnumDef::cast(node.clone()) {
        let module = module_from_child_node(db, file_id, e.syntax())?;
        Some(enum_from_module(db, module, &e).resolver(db))
    } else if node.kind() == FN_DEF || node.kind() == CONST_DEF || node.kind() == STATIC_DEF {
        Some(def_with_body_from_child_node(db, file_id, node)?.resolver(db))
    } else {
        // FIXME add missing cases
        None
    }
}

fn def_with_body_from_child_node(
    db: &impl HirDatabase,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<DefWithBody> {
    let module = module_from_child_node(db, file_id, node)?;
    let ctx = LocationCtx::new(db, module, file_id.into());
    node.ancestors().find_map(|node| {
        if let Some(def) = ast::FnDef::cast(node.clone()) {
            return Some(Function { id: ctx.to_def(&def) }.into());
        }
        if let Some(def) = ast::ConstDef::cast(node.clone()) {
            return Some(Const { id: ctx.to_def(&def) }.into());
        }
        if let Some(def) = ast::StaticDef::cast(node.clone()) {
            return Some(Static { id: ctx.to_def(&def) }.into());
        }
        None
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
    AssocItem(crate::ImplItem),
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
                Some(offset) => scope_for_offset(&scopes, &source_map, offset),
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

    pub fn resolve_struct_literal(&self, struct_lit: &ast::StructLit) -> Option<crate::VariantDef> {
        let expr_id = self.body_source_map.as_ref()?.node_expr(&struct_lit.clone().into())?;
        self.infer.as_ref()?.variant_resolution_for_expr(expr_id)
    }

    pub fn resolve_struct_pattern(&self, struct_pat: &ast::StructPat) -> Option<crate::VariantDef> {
        let pat_id = self.body_source_map.as_ref()?.node_pat(&struct_pat.clone().into())?;
        self.infer.as_ref()?.variant_resolution_for_pat(pat_id)
    }

    pub fn resolve_macro_call(
        &self,
        db: &impl HirDatabase,
        macro_call: &ast::MacroCall,
    ) -> Option<MacroDef> {
        let path = macro_call.path().and_then(Path::from_ast)?;
        self.resolver.resolve_path_as_macro(db, &path)
    }

    pub fn resolve_hir_path(
        &self,
        db: &impl HirDatabase,
        path: &crate::Path,
    ) -> PerNs<crate::Resolution> {
        self.resolver.resolve_path_without_assoc_items(db, path)
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
        let hir_path = crate::Path::from_ast(path.clone())?;
        let res = self.resolver.resolve_path_without_assoc_items(db, &hir_path);
        let res = res.clone().take_types().or_else(|| res.take_values())?;
        let res = match res {
            crate::Resolution::Def(it) => PathResolution::Def(it),
            crate::Resolution::LocalBinding(it) => {
                // We get a `PatId` from resolver, but it actually can only
                // point at `BindPat`, and not at the arbitrary pattern.
                let pat_ptr = self
                    .body_source_map
                    .as_ref()?
                    .pat_syntax(it)?
                    .map_a(|ptr| ptr.cast::<ast::BindPat>().unwrap());
                PathResolution::LocalBinding(pat_ptr)
            }
            crate::Resolution::GenericParam(it) => PathResolution::GenericParam(it),
            crate::Resolution::SelfType(it) => PathResolution::SelfType(it),
        };
        Some(res)
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
                ptr: source_map.pat_syntax(entry.pat())?,
            })
        })
    }

    pub fn all_names(&self, db: &impl HirDatabase) -> FxHashMap<Name, PerNs<crate::Resolution>> {
        self.resolver.all_names(db)
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
        callback: impl FnMut(&Ty, Function) -> Option<T>,
    ) -> Option<T> {
        // There should be no inference vars in types passed here
        // FIXME check that?
        let canonical = crate::ty::Canonical { value: ty, num_vars: 0 };
        crate::ty::method_resolution::iterate_method_candidates(
            &canonical,
            db,
            &self.resolver,
            name,
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
        .map(|it| SyntaxNodePtr::new(&it))
        .filter_map(|ptr| source_map.syntax_expr(ptr))
        .find_map(|it| scopes.scope_for(it))
}

fn scope_for_offset(
    scopes: &ExprScopes,
    source_map: &BodySourceMap,
    offset: TextUnit,
) -> Option<ScopeId> {
    scopes
        .scope_by_expr()
        .iter()
        .filter_map(|(id, scope)| Some((source_map.expr_syntax(*id)?, scope)))
        // find containing scope
        .min_by_key(|(ptr, _scope)| {
            (!(ptr.range().start() <= offset && offset <= ptr.range().end()), ptr.range().len())
        })
        .map(|(ptr, scope)| adjust(scopes, source_map, ptr, offset).unwrap_or(*scope))
}

// XXX: during completion, cursor might be outside of any particular
// expression. Try to figure out the correct scope...
fn adjust(
    scopes: &ExprScopes,
    source_map: &BodySourceMap,
    ptr: SyntaxNodePtr,
    offset: TextUnit,
) -> Option<ScopeId> {
    let r = ptr.range();
    let child_scopes = scopes
        .scope_by_expr()
        .iter()
        .filter_map(|(id, scope)| Some((source_map.expr_syntax(*id)?, scope)))
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
