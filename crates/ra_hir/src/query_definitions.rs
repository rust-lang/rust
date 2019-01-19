use std::sync::Arc;

use ra_syntax::{SyntaxNode, TreeArc};

use crate::{
    SourceFileItems, SourceItemId, HirFileId,
    Function, ExprScopes,
    db::HirDatabase,
};

pub(super) fn expr_scopes(db: &impl HirDatabase, func: Function) -> Arc<ExprScopes> {
    let body = db.body_hir(func);
    let res = ExprScopes::new(body);
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
    let source_file = db.hir_parse(source_item_id.file_id);
    db.file_items(source_item_id.file_id)[source_item_id.item_id]
        .to_node(&source_file)
        .to_owned()
}
