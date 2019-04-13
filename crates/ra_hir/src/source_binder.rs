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
    SyntaxNode, AstPtr, TextUnit,
    ast::{self, AstNode, NameOwner},
    algo::find_node_at_offset,
};

use crate::{
    HirDatabase, Function, Struct, Enum, Const, Static, Either, DefWithBody,
    AsName, Module, HirFileId, Crate, Trait, Resolver,
    expr::scope::{ReferenceDescriptor, ScopeEntryWithSyntax},
    ids::LocationCtx,
    expr, AstId
};

/// Locates the module by `FileId`. Picks topmost module in the file.
pub fn module_from_file_id(db: &impl HirDatabase, file_id: FileId) -> Option<Module> {
    module_from_source(db, file_id.into(), None)
}

/// Locates the child module by `mod child;` declaration.
pub fn module_from_declaration(
    db: &impl HirDatabase,
    file_id: FileId,
    decl: &ast::Module,
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
    let file = db.parse(position.file_id);
    match find_node_at_offset::<ast::Module>(file.syntax(), position.offset) {
        Some(m) if !m.has_semi() => module_from_inline(db, position.file_id.into(), m),
        _ => module_from_file_id(db, position.file_id.into()),
    }
}

fn module_from_inline(
    db: &impl HirDatabase,
    file_id: FileId,
    module: &ast::Module,
) -> Option<Module> {
    assert!(!module.has_semi());
    let file_id = file_id.into();
    let ast_id_map = db.ast_id_map(file_id);
    let item_id = ast_id_map.ast_id(module).with_file_id(file_id);
    module_from_source(db, file_id, Some(item_id))
}

/// Locates the module by child syntax element within the module
pub fn module_from_child_node(
    db: &impl HirDatabase,
    file_id: FileId,
    child: &SyntaxNode,
) -> Option<Module> {
    if let Some(m) = child.ancestors().filter_map(ast::Module::cast).find(|it| !it.has_semi()) {
        module_from_inline(db, file_id.into(), m)
    } else {
        module_from_file_id(db, file_id.into())
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

fn function_from_source(
    db: &impl HirDatabase,
    file_id: FileId,
    fn_def: &ast::FnDef,
) -> Option<Function> {
    let module = module_from_child_node(db, file_id, fn_def.syntax())?;
    let res = function_from_module(db, module, fn_def);
    Some(res)
}

fn function_from_module(db: &impl HirDatabase, module: Module, fn_def: &ast::FnDef) -> Function {
    let (file_id, _) = module.definition_source(db);
    let file_id = file_id.into();
    let ctx = LocationCtx::new(db, module, file_id);
    Function { id: ctx.to_def(fn_def) }
}

fn function_from_child_node(
    db: &impl HirDatabase,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<Function> {
    let fn_def = node.ancestors().find_map(ast::FnDef::cast)?;
    function_from_source(db, file_id, fn_def)
}

pub fn struct_from_module(
    db: &impl HirDatabase,
    module: Module,
    struct_def: &ast::StructDef,
) -> Struct {
    let (file_id, _) = module.definition_source(db);
    let file_id = file_id.into();
    let ctx = LocationCtx::new(db, module, file_id);
    Struct { id: ctx.to_def(struct_def) }
}

pub fn enum_from_module(db: &impl HirDatabase, module: Module, enum_def: &ast::EnumDef) -> Enum {
    let (file_id, _) = module.definition_source(db);
    let file_id = file_id.into();
    let ctx = LocationCtx::new(db, module, file_id);
    Enum { id: ctx.to_def(enum_def) }
}

pub fn trait_from_module(
    db: &impl HirDatabase,
    module: Module,
    trait_def: &ast::TraitDef,
) -> Trait {
    let (file_id, _) = module.definition_source(db);
    let file_id = file_id.into();
    let ctx = LocationCtx::new(db, module, file_id);
    Trait { id: ctx.to_def(trait_def) }
}

fn resolver_for_node(
    db: &impl HirDatabase,
    file_id: FileId,
    node: &SyntaxNode,
    offset: Option<TextUnit>,
) -> Resolver {
    node.ancestors()
        .find_map(|node| {
            if ast::Expr::cast(node).is_some() || ast::Block::cast(node).is_some() {
                if let Some(func) = function_from_child_node(db, file_id, node) {
                    let scopes = func.scopes(db);
                    let scope = match offset {
                        None => scopes.scope_for(&node),
                        Some(offset) => scopes.scope_for_offset(offset),
                    };
                    Some(expr::resolver_for_scope(func.body(db), db, scope))
                } else {
                    // FIXME const/static/array length
                    None
                }
            } else {
                try_get_resolver_for_node(db, file_id, node)
            }
        })
        .unwrap_or_default()
}

fn try_get_resolver_for_node(
    db: &impl HirDatabase,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<Resolver> {
    if let Some(module) = ast::Module::cast(node) {
        Some(module_from_declaration(db, file_id, module)?.resolver(db))
    } else if let Some(_) = ast::SourceFile::cast(node) {
        Some(module_from_source(db, file_id.into(), None)?.resolver(db))
    } else if let Some(s) = ast::StructDef::cast(node) {
        let module = module_from_child_node(db, file_id, s.syntax())?;
        Some(struct_from_module(db, module, s).resolver(db))
    } else if let Some(e) = ast::EnumDef::cast(node) {
        let module = module_from_child_node(db, file_id, e.syntax())?;
        Some(enum_from_module(db, module, e).resolver(db))
    } else if let Some(f) = ast::FnDef::cast(node) {
        function_from_source(db, file_id, f).map(|f| f.resolver(db))
    } else {
        // FIXME add missing cases
        None
    }
}

pub fn def_with_body_from_child_node(
    db: &impl HirDatabase,
    file_id: FileId,
    node: &SyntaxNode,
) -> Option<DefWithBody> {
    let module = module_from_child_node(db, file_id, node)?;
    let ctx = LocationCtx::new(db, module, file_id.into());
    node.ancestors().find_map(|node| {
        if let Some(def) = ast::FnDef::cast(node) {
            return Some(Function { id: ctx.to_def(def) }.into());
        }
        if let Some(def) = ast::ConstDef::cast(node) {
            return Some(Const { id: ctx.to_def(def) }.into());
        }
        if let Some(def) = ast::StaticDef::cast(node) {
            return Some(Static { id: ctx.to_def(def) }.into());
        }
        None
    })
}

/// `SourceAnalyzer` is a convenience wrapper which exposes HIR API in terms of
/// original source files. It should not be used inside the HIR itself.
#[derive(Debug)]
pub struct SourceAnalyzer {
    resolver: Resolver,
    body_source_map: Option<Arc<crate::expr::BodySourceMap>>,
    infer: Option<Arc<crate::ty::InferenceResult>>,
    scopes: Option<crate::expr::ScopesWithSourceMap>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathResolution {
    /// An item
    Def(crate::ModuleDef),
    /// A local binding (only value namespace)
    LocalBinding(Either<AstPtr<ast::Pat>, AstPtr<ast::SelfParam>>),
    /// A generic parameter
    GenericParam(u32),
    SelfType(crate::ImplBlock),
    AssocItem(crate::ImplItem),
}

impl SourceAnalyzer {
    pub fn new(
        db: &impl HirDatabase,
        file_id: FileId,
        node: &SyntaxNode,
        offset: Option<TextUnit>,
    ) -> SourceAnalyzer {
        let def_with_body = def_with_body_from_child_node(db, file_id, node);
        SourceAnalyzer {
            resolver: resolver_for_node(db, file_id, node, offset),
            body_source_map: def_with_body.map(|it| it.body_source_map(db)),
            infer: def_with_body.map(|it| it.infer(db)),
            scopes: def_with_body.map(|it| it.scopes(db)),
        }
    }

    pub fn resolver(&self) -> &Resolver {
        &self.resolver
    }

    pub fn type_of(&self, _db: &impl HirDatabase, expr: &ast::Expr) -> Option<crate::Ty> {
        let expr_id = self.body_source_map.as_ref()?.node_expr(expr)?;
        Some(self.infer.as_ref()?[expr_id].clone())
    }

    pub fn type_of_pat(&self, _db: &impl HirDatabase, pat: &ast::Pat) -> Option<crate::Ty> {
        let pat_id = self.body_source_map.as_ref()?.node_pat(pat)?;
        Some(self.infer.as_ref()?[pat_id].clone())
    }

    pub fn resolve_method_call(&self, call: &ast::MethodCallExpr) -> Option<Function> {
        let expr_id = self.body_source_map.as_ref()?.node_expr(call.into())?;
        self.infer.as_ref()?.method_resolution(expr_id)
    }

    pub fn resolve_field(&self, field: &ast::FieldExpr) -> Option<crate::StructField> {
        let expr_id = self.body_source_map.as_ref()?.node_expr(field.into())?;
        self.infer.as_ref()?.field_resolution(expr_id)
    }

    pub fn resolve_path(&self, db: &impl HirDatabase, path: &ast::Path) -> Option<PathResolution> {
        if let Some(path_expr) = path.syntax().parent().and_then(ast::PathExpr::cast) {
            let expr_id = self.body_source_map.as_ref()?.node_expr(path_expr.into())?;
            if let Some(assoc) = self.infer.as_ref()?.assoc_resolutions_for_expr(expr_id) {
                return Some(PathResolution::AssocItem(assoc));
            }
        }
        if let Some(path_pat) = path.syntax().parent().and_then(ast::PathPat::cast) {
            let pat_id = self.body_source_map.as_ref()?.node_pat(path_pat.into())?;
            if let Some(assoc) = self.infer.as_ref()?.assoc_resolutions_for_pat(pat_id) {
                return Some(PathResolution::AssocItem(assoc));
            }
        }
        let hir_path = crate::Path::from_ast(path)?;
        let res = self.resolver.resolve_path(db, &hir_path);
        let res = res.clone().take_types().or_else(|| res.take_values())?;
        let res = match res {
            crate::Resolution::Def(it) => PathResolution::Def(it),
            crate::Resolution::LocalBinding(it) => {
                PathResolution::LocalBinding(self.body_source_map.as_ref()?.pat_syntax(it)?)
            }
            crate::Resolution::GenericParam(it) => PathResolution::GenericParam(it),
            crate::Resolution::SelfType(it) => PathResolution::SelfType(it),
        };
        Some(res)
    }

    pub fn find_all_refs(&self, pat: &ast::BindPat) -> Option<Vec<ReferenceDescriptor>> {
        self.scopes.as_ref().map(|it| it.find_all_refs(pat))
    }

    pub fn resolve_local_name(&self, name_ref: &ast::NameRef) -> Option<ScopeEntryWithSyntax> {
        self.scopes.as_ref()?.resolve_local_name(name_ref)
    }

    #[cfg(test)]
    pub(crate) fn body_source_map(&self) -> Arc<crate::expr::BodySourceMap> {
        self.body_source_map.clone().unwrap()
    }

    #[cfg(test)]
    pub(crate) fn inference_result(&self) -> Arc<crate::ty::InferenceResult> {
        self.infer.clone().unwrap()
    }
}
