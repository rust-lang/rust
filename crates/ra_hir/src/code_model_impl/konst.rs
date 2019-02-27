use std::sync::Arc;

use ra_syntax::ast::{NameOwner, TypeAscriptionOwner};

use crate::{
    Name, AsName, Const, ConstSignature, Static,
    type_ref::{TypeRef},
    PersistentHirDatabase,
};

fn const_signature_for<N: NameOwner + TypeAscriptionOwner>(node: &N) -> Arc<ConstSignature> {
    let name = node.name().map(|n| n.as_name()).unwrap_or_else(Name::missing);
    let type_ref = TypeRef::from_ast_opt(node.ascribed_type());
    let sig = ConstSignature { name, type_ref };
    Arc::new(sig)
}

impl ConstSignature {
    pub(crate) fn const_signature_query(
        db: &impl PersistentHirDatabase,
        konst: Const,
    ) -> Arc<ConstSignature> {
        let (_, node) = konst.source(db);
        const_signature_for(&*node)
    }

    pub(crate) fn static_signature_query(
        db: &impl PersistentHirDatabase,
        konst: Static,
    ) -> Arc<ConstSignature> {
        let (_, node) = konst.source(db);
        const_signature_for(&*node)
    }
}
