use std::sync::Arc;

use ra_syntax::{
    ast::{AstNode, FnDef, FnDefNode},
};

use crate::{
    descriptors::{
        DescriptorDatabase,
        function::{FnId, FnScopes},
    },
};

/// Resolve `FnId` to the corresponding `SyntaxNode`
/// TODO: this should return something more type-safe then `SyntaxNode`
pub(crate) fn fn_syntax(db: &impl DescriptorDatabase, fn_id: FnId) -> FnDefNode {
    let syntax = db.resolve_syntax_ptr(fn_id.0);
    let fn_def = FnDef::cast(syntax.borrowed()).unwrap();
    FnDefNode::new(fn_def)
}

pub(crate) fn fn_scopes(db: &impl DescriptorDatabase, fn_id: FnId) -> Arc<FnScopes> {
    let syntax = db.fn_syntax(fn_id);
    let res = FnScopes::new(syntax.ast());
    Arc::new(res)
}
