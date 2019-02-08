use std::sync::Arc;

use ra_syntax::ast::{self, NameOwner};

use crate::{
    HirDatabase, Name, AsName, Function, FnSignature,
    type_ref::{TypeRef, Mutability},
    PersistentHirDatabase,
    impl_block::ImplBlock,
};

impl Function {
    // TODO impl_block should probably also be part of the code model API?

    /// The containing impl block, if this is a method.
    pub(crate) fn impl_block(&self, db: &impl HirDatabase) -> Option<ImplBlock> {
        let module_impls = db.impls_in_module(self.module(db));
        ImplBlock::containing(module_impls, (*self).into())
    }
}

impl FnSignature {
    pub(crate) fn fn_signature_query(
        db: &impl PersistentHirDatabase,
        func: Function,
    ) -> Arc<FnSignature> {
        let (_, node) = func.source(db);
        let name = node.name().map(|n| n.as_name()).unwrap_or_else(Name::missing);
        let mut params = Vec::new();
        let mut has_self_param = false;
        if let Some(param_list) = node.param_list() {
            if let Some(self_param) = param_list.self_param() {
                let self_type = if let Some(type_ref) = self_param.type_ref() {
                    TypeRef::from_ast(type_ref)
                } else {
                    let self_type = TypeRef::Path(Name::self_type().into());
                    match self_param.flavor() {
                        ast::SelfParamFlavor::Owned => self_type,
                        ast::SelfParamFlavor::Ref => {
                            TypeRef::Reference(Box::new(self_type), Mutability::Shared)
                        }
                        ast::SelfParamFlavor::MutRef => {
                            TypeRef::Reference(Box::new(self_type), Mutability::Mut)
                        }
                    }
                };
                params.push(self_type);
                has_self_param = true;
            }
            for param in param_list.params() {
                let type_ref = TypeRef::from_ast_opt(param.type_ref());
                params.push(type_ref);
            }
        }
        let ret_type = if let Some(type_ref) = node.ret_type().and_then(|rt| rt.type_ref()) {
            TypeRef::from_ast(type_ref)
        } else {
            TypeRef::unit()
        };

        let sig = FnSignature { name, params, ret_type, has_self_param };
        Arc::new(sig)
    }
}
