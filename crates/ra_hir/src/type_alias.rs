//! HIR for type aliases (i.e. the `type` keyword).

use std::sync::Arc;

use hir_def::type_ref::TypeRef;
use hir_expand::name::{AsName, Name};

use ra_syntax::ast::NameOwner;

use crate::{
    db::{AstDatabase, DefDatabase},
    HasSource, TypeAlias,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAliasData {
    pub(crate) name: Name,
    pub(crate) type_ref: Option<TypeRef>,
}

impl TypeAliasData {
    pub(crate) fn type_alias_data_query(
        db: &(impl DefDatabase + AstDatabase),
        typ: TypeAlias,
    ) -> Arc<TypeAliasData> {
        let node = typ.source(db).ast;
        let name = node.name().map_or_else(Name::missing, |n| n.as_name());
        let type_ref = node.type_ref().map(TypeRef::from_ast);
        Arc::new(TypeAliasData { name, type_ref })
    }
}
