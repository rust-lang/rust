use std::{
    sync::Arc,
    time::Instant,
};

use rustc_hash::FxHashMap;
use ra_syntax::{
    AstNode, SyntaxNode, TreeArc,
};
use ra_db::{CrateId};

use crate::{
    SourceFileItems, SourceItemId, HirFileId,
    Function, FnScopes, Module,
    db::HirDatabase,
    nameres::{ItemMap, Resolver},
};

pub(super) fn fn_scopes(db: &impl HirDatabase, func: Function) -> Arc<FnScopes> {
    let body = db.body_hir(func);
    let res = FnScopes::new(body);
    Arc::new(res)
}

pub(super) fn file_items(db: &impl HirDatabase, file_id: HirFileId) -> Arc<SourceFileItems> {
    let source_file = db.hir_parse(file_id);
    let res = SourceFileItems::new(file_id, &source_file);
    Arc::new(res)
}

pub(super) fn file_item(
    db: &impl HirDatabase,
    source_item_id: SourceItemId,
) -> TreeArc<SyntaxNode> {
    match source_item_id.item_id {
        Some(id) => db.file_items(source_item_id.file_id)[id].to_owned(),
        None => db.hir_parse(source_item_id.file_id).syntax().to_owned(),
    }
}

pub(super) fn item_map(db: &impl HirDatabase, crate_id: CrateId) -> Arc<ItemMap> {
    let start = Instant::now();
    let module_tree = db.module_tree(crate_id);
    let input = module_tree
        .modules()
        .map(|module_id| {
            (
                module_id,
                db.lower_module_module(Module {
                    krate: crate_id,
                    module_id,
                }),
            )
        })
        .collect::<FxHashMap<_, _>>();

    let resolver = Resolver::new(db, &input, crate_id);
    let res = resolver.resolve();
    let elapsed = start.elapsed();
    log::info!("item_map: {:?}", elapsed);
    Arc::new(res)
}
