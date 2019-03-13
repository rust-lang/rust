/// Lookup hir elements using positions in the source code. This is a lossy
/// transformation: in general, a single source might correspond to several
/// modules, functions, etc, due to macros, cfgs and `#[path=]` attributes on
/// modules.
///
/// So, this modules should not be used during hir construction, it exists
/// purely for "IDE needs".
use ra_db::{FileId, FilePosition};
use ra_syntax::{
    SmolStr, TextRange, SyntaxNode,
    ast::{self, AstNode, NameOwner},
    algo::{find_node_at_offset, find_leaf_at_offset},
};

use crate::{
    HirDatabase, Function, ModuleDef, Struct, Enum,
    AsName, Module, HirFileId, Crate, Trait, Resolver,
    ids::{LocationCtx, SourceFileItemId},
    expr
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
    let file_items = db.file_items(file_id);
    let item_id = file_items.id_of(file_id, module.syntax());
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
    decl_id: Option<SourceFileItemId>,
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

pub fn function_from_position(db: &impl HirDatabase, position: FilePosition) -> Option<Function> {
    let file = db.parse(position.file_id);
    let fn_def = find_node_at_offset::<ast::FnDef>(file.syntax(), position.offset)?;
    function_from_source(db, position.file_id, fn_def)
}

pub fn function_from_source(
    db: &impl HirDatabase,
    file_id: FileId,
    fn_def: &ast::FnDef,
) -> Option<Function> {
    let module = module_from_child_node(db, file_id, fn_def.syntax())?;
    let res = function_from_module(db, module, fn_def);
    Some(res)
}

pub fn function_from_module(
    db: &impl HirDatabase,
    module: Module,
    fn_def: &ast::FnDef,
) -> Function {
    let (file_id, _) = module.definition_source(db);
    let file_id = file_id.into();
    let ctx = LocationCtx::new(db, module, file_id);
    Function { id: ctx.to_def(fn_def) }
}

pub fn function_from_child_node(
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

pub fn macro_symbols(db: &impl HirDatabase, file_id: FileId) -> Vec<(SmolStr, TextRange)> {
    let module = match module_from_file_id(db, file_id) {
        Some(it) => it,
        None => return Vec::new(),
    };
    let items = db.lower_module(module);
    let mut res = Vec::new();

    for macro_call_id in items
        .declarations
        .iter()
        .filter_map(|(_, it)| it.clone().take_types())
        .filter_map(|it| match it {
            ModuleDef::Trait(it) => Some(it),
            _ => None,
        })
        .filter_map(|it| it.source(db).0.as_macro_call_id())
    {
        if let Some(exp) = db.expand_macro_invocation(macro_call_id) {
            let loc = macro_call_id.loc(db);
            let syntax = db.file_item(loc.source_item_id);
            let macro_call = ast::MacroCall::cast(&syntax).unwrap();
            let off = macro_call.token_tree().unwrap().syntax().range().start();
            let file = exp.file();
            for trait_def in file.syntax().descendants().filter_map(ast::TraitDef::cast) {
                if let Some(name) = trait_def.name() {
                    let dst_range = name.syntax().range();
                    if let Some(src_range) = exp.map_range_back(dst_range) {
                        res.push((name.text().clone(), src_range + off))
                    }
                }
            }
        }
    }

    res
}

pub fn resolver_for_position(db: &impl HirDatabase, position: FilePosition) -> Resolver {
    let file_id = position.file_id;
    let file = db.parse(file_id);
    find_leaf_at_offset(file.syntax(), position.offset)
        .find_map(|node| {
            node.ancestors().find_map(|node| {
                if ast::Expr::cast(node).is_some() || ast::Block::cast(node).is_some() {
                    if let Some(func) = function_from_child_node(db, file_id, node) {
                        let scopes = func.scopes(db);
                        let scope = scopes.scope_for_offset(position.offset);
                        Some(expr::resolver_for_scope(func.body(db), db, scope))
                    } else {
                        // TODO const/static/array length
                        None
                    }
                } else {
                    try_get_resolver_for_node(db, file_id, node)
                }
            })
        })
        .unwrap_or_default()
}

pub fn resolver_for_node(db: &impl HirDatabase, file_id: FileId, node: &SyntaxNode) -> Resolver {
    node.ancestors()
        .find_map(|node| {
            if ast::Expr::cast(node).is_some() || ast::Block::cast(node).is_some() {
                if let Some(func) = function_from_child_node(db, file_id, node) {
                    let scopes = func.scopes(db);
                    let scope = scopes.scope_for(&node);
                    Some(expr::resolver_for_scope(func.body(db), db, scope))
                } else {
                    // TODO const/static/array length
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
        // TODO add missing cases
        None
    }
}
