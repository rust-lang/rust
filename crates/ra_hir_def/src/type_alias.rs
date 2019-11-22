//! HIR for type aliases (i.e. the `type` keyword).

use std::sync::Arc;

use hir_expand::name::{AsName, Name};

use ra_syntax::ast::NameOwner;

use crate::{db::DefDatabase2, type_ref::TypeRef, HasSource, Lookup, TypeAliasId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAliasData {
    pub name: Name,
    pub type_ref: Option<TypeRef>,
}

impl TypeAliasData {
    pub(crate) fn type_alias_data_query(
        db: &impl DefDatabase2,
        typ: TypeAliasId,
    ) -> Arc<TypeAliasData> {
        let node = typ.lookup(db).source(db).value;
        let name = node.name().map_or_else(Name::missing, |n| n.as_name());
        let type_ref = node.type_ref().map(TypeRef::from_ast);
        Arc::new(TypeAliasData { name, type_ref })
    }
}
