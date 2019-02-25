use std::sync::Arc;

use ra_syntax::ast::{self, NameOwner};

use crate::{
    Name, AsName, Const, ConstSignature, Static,
    type_ref::{TypeRef},
    PersistentHirDatabase,
};

fn const_signature_for<N: NameOwner>(
    node: &N,
    type_ref: Option<&ast::TypeRef>,
) -> Arc<ConstSignature> {
    let name = node.name().map(|n| n.as_name()).unwrap_or_else(Name::missing);
    let type_ref = TypeRef::from_ast_opt(type_ref);
    let sig = ConstSignature { name, type_ref };
    Arc::new(sig)
}

impl ConstSignature {
    pub(crate) fn const_signature_query(
        db: &impl PersistentHirDatabase,
        konst: Const,
    ) -> Arc<ConstSignature> {
        let (_, node) = konst.source(db);
        const_signature_for(&*node, node.type_ref())
    }

    pub(crate) fn static_signature_query(
        db: &impl PersistentHirDatabase,
        konst: Static,
    ) -> Arc<ConstSignature> {
        let (_, node) = konst.source(db);
        const_signature_for(&*node, node.type_ref())
    }
}
