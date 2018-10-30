use ra_editor::{CompletionItem, find_node_at_offset};
use ra_syntax::{
    AtomEdit, File, TextUnit, AstNode,
    ast::{self, ModuleItemOwner, AstChildren},
};

use crate::{
    FileId, Cancelable,
    input::FilesDatabase,
    db::{self, SyntaxDatabase},
    descriptors::module::{ModulesDatabase, ModuleTree, ModuleId, scope::ModuleScope},
};

pub(crate) fn resolve_based_completion(db: &db::RootDatabase, file_id: FileId, offset: TextUnit) -> Cancelable<Option<Vec<CompletionItem>>> {
    let source_root_id = db.file_source_root(file_id);
    let file = db.file_syntax(file_id);
    let module_tree = db.module_tree(source_root_id)?;
    let module_id = match module_tree.any_module_for_file(file_id) {
        None => return Ok(None),
        Some(it) => it,
    };
    let file = {
        let edit = AtomEdit::insert(offset, "intellijRulezz".to_string());
        file.reparse(&edit)
    };
    let target_module_id = match find_target_module(&module_tree, module_id, &file, offset) {
        None => return Ok(None),
        Some(it) => it,
    };
    let module_scope = db.module_scope(source_root_id, target_module_id)?;
    let res: Vec<_> = module_scope
        .entries()
        .iter()
        .map(|entry| CompletionItem {
            label: entry.name().to_string(),
            lookup: None,
            snippet: None,
        })
        .collect();
    Ok(Some(res))
}



pub(crate) fn find_target_module(module_tree: &ModuleTree, module_id: ModuleId, file: &File, offset: TextUnit) -> Option<ModuleId> {
    let name_ref: ast::NameRef = find_node_at_offset(file.syntax(), offset)?;
    let mut crate_path = crate_path(name_ref)?;

    crate_path.pop();
    let mut target_module = module_id.root(&module_tree);
    for name in crate_path {
        target_module = target_module.child(module_tree, name.text().as_str())?;
    }
    Some(target_module)
}

fn crate_path(name_ref: ast::NameRef) -> Option<Vec<ast::NameRef>> {
    let mut path = name_ref.syntax()
        .parent().and_then(ast::PathSegment::cast)?
        .parent_path();
    let mut res = Vec::new();
    loop {
        let segment = path.segment()?;
        match segment.kind()? {
            ast::PathSegmentKind::Name(name) => res.push(name),
            ast::PathSegmentKind::CrateKw => break,
            ast::PathSegmentKind::SelfKw | ast::PathSegmentKind::SuperKw =>
                return None,
        }
        path = path.qualifier()?;
    }
    res.reverse();
    Some(res)
}
