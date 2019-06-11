//! HIR for type aliases (i.e. the `type` keyword).

use std::sync::Arc;

use crate::{TypeAlias, DefDatabase, AstDatabase, type_ref::TypeRef};

pub(crate) fn type_alias_ref_query(
    db: &(impl DefDatabase + AstDatabase),
    typ: TypeAlias,
) -> Arc<TypeRef> {
    let node = typ.source(db).ast;
    Arc::new(TypeRef::from_ast_opt(node.type_ref()))
}
