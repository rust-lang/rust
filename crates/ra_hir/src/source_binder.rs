/// Lookup hir elements using position in the source code. This is a lossy
/// transformation: in general, a single source might correspond to several
/// modules, functions, etc, due to macros, cfgs and `#[path=]` attributes on
/// modules.
///
/// So, this modules should not be used during hir construction, it exists
/// purely for "IDE needs".
use ra_db::{FileId, FilePosition, Cancelable};
use ra_editor::find_node_at_offset;
use ra_syntax::{
    ast::{self, AstNode},
    SyntaxNodeRef,
};

use crate::{
    HirDatabase, Module, Function, SourceItemId,
    module::ModuleSource,
    DefKind, DefLoc
};

/// Locates the module by `FileId`. Picks topmost module in the file.
pub fn module_from_file_id(db: &impl HirDatabase, file_id: FileId) -> Cancelable<Option<Module>> {
    let module_source = ModuleSource::new_file(file_id);
    module_from_source(db, module_source)
}

/// Locates the module by position in the source code.
pub fn module_from_position(
    db: &impl HirDatabase,
    position: FilePosition,
) -> Cancelable<Option<Module>> {
    let file = db.source_file(position.file_id);
    let module_source = match find_node_at_offset::<ast::Module>(file.syntax(), position.offset) {
        Some(m) if !m.has_semi() => ModuleSource::new_inline(db, position.file_id, m),
        _ => ModuleSource::new_file(position.file_id),
    };
    module_from_source(db, module_source)
}

/// Locates the module by child syntax element within the module
pub fn module_from_child_node(
    db: &impl HirDatabase,
    file_id: FileId,
    child: SyntaxNodeRef,
) -> Cancelable<Option<Module>> {
    let module_source = if let Some(m) = child
        .ancestors()
        .filter_map(ast::Module::cast)
        .find(|it| !it.has_semi())
    {
        ModuleSource::new_inline(db, file_id, m)
    } else {
        ModuleSource::new_file(file_id)
    };
    module_from_source(db, module_source)
}

fn module_from_source(
    db: &impl HirDatabase,
    module_source: ModuleSource,
) -> Cancelable<Option<Module>> {
    let source_root_id = db.file_source_root(module_source.file_id());
    let module_tree = db.module_tree(source_root_id)?;
    let m = module_tree
        .modules_with_sources()
        .find(|(_id, src)| src == &module_source);
    let module_id = ctry!(m).0;
    Ok(Some(Module::new(db, source_root_id, module_id)?))
}

pub fn function_from_source(
    db: &impl HirDatabase,
    file_id: FileId,
    fn_def: ast::FnDef,
) -> Cancelable<Option<Function>> {
    let module = ctry!(module_from_child_node(db, file_id, fn_def.syntax())?);
    let file_items = db.file_items(file_id);
    let item_id = file_items.id_of(file_id, fn_def.syntax());
    let source_item_id = SourceItemId {
        file_id,
        item_id: Some(item_id),
    };
    let def_loc = DefLoc {
        kind: DefKind::Function,
        source_root_id: module.source_root_id,
        module_id: module.module_id,
        source_item_id,
    };
    Ok(Some(Function::new(def_loc.id(db))))
}

pub fn function_from_child_node(
    db: &impl HirDatabase,
    file_id: FileId,
    node: SyntaxNodeRef,
) -> Cancelable<Option<Function>> {
    let fn_def = ctry!(node.ancestors().find_map(ast::FnDef::cast));
    function_from_source(db, file_id, fn_def)
}
