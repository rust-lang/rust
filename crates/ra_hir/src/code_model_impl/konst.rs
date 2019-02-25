use std::sync::Arc;

use ra_syntax::ast::{NameOwner};

use crate::{
    Name, AsName, Const, ConstSignature,
    type_ref::{TypeRef},
    PersistentHirDatabase,
};

impl ConstSignature {
    pub(crate) fn const_signature_query(
        db: &impl PersistentHirDatabase,
        konst: Const,
    ) -> Arc<ConstSignature> {
        let (_, node) = konst.source(db);

        let name = node.name().map(|n| n.as_name()).unwrap_or_else(Name::missing);

        let type_ref = TypeRef::from_ast_opt(node.type_ref());

        let sig = ConstSignature { name, type_ref };

        Arc::new(sig)
    }
}
