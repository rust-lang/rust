use std::{
    sync::Arc,
    time::Instant,
};

use rustc_hash::FxHashMap;
use ra_syntax::{
    AstNode, SyntaxNode, TreeArc,
};
use ra_db::SourceRootId;

use crate::{
    SourceFileItems, SourceItemId, DefId, HirFileId,
    FnScopes,
    db::HirDatabase,
    nameres::{ItemMap, Resolver},
};

pub(super) fn fn_scopes(db: &impl HirDatabase, def_id: DefId) -> Arc<FnScopes> {
    let body = db.body_hir(def_id);
    let res = FnScopes::new(body);
    Arc::new(res)
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

pub(super) fn item_map(db: &impl HirDatabase, source_root: SourceRootId) -> Arc<ItemMap> {
    let start = Instant::now();
    let module_tree = db.module_tree(source_root);
    let input = module_tree
        .modules()
        .map(|id| (id, db.lower_module_module(source_root, id)))
        .collect::<FxHashMap<_, _>>();

    let resolver = Resolver::new(db, &input, source_root, module_tree);
    let res = resolver.resolve();
    let elapsed = start.elapsed();
    log::info!("item_map: {:?}", elapsed);
    Arc::new(res)
}
