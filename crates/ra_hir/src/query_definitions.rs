use std::{
    sync::Arc,
    time::Instant,
};

use rustc_hash::FxHashMap;
use ra_syntax::{
    AstNode, SyntaxNode, TreeArc,
    ast::{self, ModuleItemOwner}
};
use ra_db::{SourceRootId, Cancelable,};

use crate::{
    SourceFileItems, SourceItemId, DefId, HirFileId, ModuleSource,
    MacroCallLoc, FnScopes,
    db::HirDatabase,
    module_tree::ModuleId,
    nameres::{InputModuleItems, ItemMap, Resolver},
};

pub(super) fn fn_scopes(db: &impl HirDatabase, def_id: DefId) -> Cancelable<Arc<FnScopes>> {
    let body = db.body_hir(def_id)?;
    let res = FnScopes::new(body);
    Ok(Arc::new(res))
}

pub(super) fn file_items(db: &impl HirDatabase, file_id: HirFileId) -> Arc<SourceFileItems> {
    let source_file = db.hir_source_file(file_id);
    let res = SourceFileItems::new(file_id, &source_file);
    Arc::new(res)
}

pub(super) fn file_item(
    db: &impl HirDatabase,
    source_item_id: SourceItemId,
) -> TreeArc<SyntaxNode> {
    match source_item_id.item_id {
        Some(id) => db.file_items(source_item_id.file_id)[id].to_owned(),
        None => db
            .hir_source_file(source_item_id.file_id)
            .syntax()
            .to_owned(),
    }
}

pub(super) fn input_module_items(
    db: &impl HirDatabase,
    source_root_id: SourceRootId,
    module_id: ModuleId,
) -> Cancelable<Arc<InputModuleItems>> {
    let module_tree = db.module_tree(source_root_id);
    let source = module_id.source(&module_tree);
    let file_id = source.file_id;
    let source = ModuleSource::from_source_item_id(db, source);
    let file_items = db.file_items(file_id);
    let fill = |acc: &mut InputModuleItems, items: &mut Iterator<Item = ast::ItemOrMacro>| {
        for item in items {
            match item {
                ast::ItemOrMacro::Item(it) => {
                    acc.add_item(file_id, &file_items, it);
                }
                ast::ItemOrMacro::Macro(macro_call) => {
                    let item_id = file_items.id_of_unchecked(macro_call.syntax());
                    let loc = MacroCallLoc {
                        source_root_id,
                        module_id,
                        source_item_id: SourceItemId {
                            file_id,
                            item_id: Some(item_id),
                        },
                    };
                    let id = loc.id(db);
                    let file_id = HirFileId::from(id);
                    let file_items = db.file_items(file_id);
                    //FIXME: expand recursively
                    for item in db.hir_source_file(file_id).items() {
                        acc.add_item(file_id, &file_items, item);
                    }
                }
            }
        }
    };

    let mut res = InputModuleItems::default();
    match source {
        ModuleSource::SourceFile(it) => fill(&mut res, &mut it.items_with_macros()),
        ModuleSource::Module(it) => {
            if let Some(item_list) = it.item_list() {
                fill(&mut res, &mut item_list.items_with_macros())
            }
        }
    };
    Ok(Arc::new(res))
}

pub(super) fn item_map(
    db: &impl HirDatabase,
    source_root: SourceRootId,
) -> Cancelable<Arc<ItemMap>> {
    let start = Instant::now();
    let module_tree = db.module_tree(source_root);
    let input = module_tree
        .modules()
        .map(|id| {
            let items = db.input_module_items(source_root, id)?;
            Ok((id, items))
        })
        .collect::<Cancelable<FxHashMap<_, _>>>()?;

    let resolver = Resolver::new(db, &input, source_root, module_tree);
    let res = resolver.resolve()?;
    let elapsed = start.elapsed();
    log::info!("item_map: {:?}", elapsed);
    Ok(Arc::new(res))
}
