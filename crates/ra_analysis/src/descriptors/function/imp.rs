use std::sync::Arc;

use ra_syntax::ast::{AstNode, FnDef, FnDefNode};

use crate::descriptors::{
    function::{FnId, FnScopes},
    DescriptorDatabase,
};

/// Resolve `FnId` to the corresponding `SyntaxNode`
pub(crate) fn fn_syntax(db: &impl DescriptorDatabase, fn_id: FnId) -> FnDefNode {
    let ptr = db.id_maps().fn_ptr(fn_id);
    let syntax = db.resolve_syntax_ptr(ptr);
    FnDef::cast(syntax.borrowed()).unwrap().owned()
}

pub(crate) fn fn_scopes(db: &impl DescriptorDatabase, fn_id: FnId) -> Arc<FnScopes> {
    let syntax = db._fn_syntax(fn_id);
    let res = FnScopes::new(syntax.borrowed());
    Arc::new(res)
}
