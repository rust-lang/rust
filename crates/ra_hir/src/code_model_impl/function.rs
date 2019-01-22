mod scope;

use std::sync::Arc;

use ra_syntax::{TreeArc, ast::{self, NameOwner, DocCommentsOwner}};

use crate::{
    DefId, HirDatabase, Name, AsName, Function, FnSignature, Module,
    type_ref::{TypeRef, Mutability},
    expr::Body,
    impl_block::ImplBlock,
    code_model_impl::def_id_to_ast,
};

pub use self::scope::{FnScopes, ScopesWithSyntaxMapping, ScopeEntryWithSyntax};

impl Function {
    pub(crate) fn new(def_id: DefId) -> Function {
        Function { def_id }
    }

    pub(crate) fn body(&self, db: &impl HirDatabase) -> Arc<Body> {
        db.body_hir(self.def_id)
    }

    pub(crate) fn module(&self, db: &impl HirDatabase) -> Module {
        self.def_id.module(db)
    }

    /// The containing impl block, if this is a method.
    pub(crate) fn impl_block(&self, db: &impl HirDatabase) -> Option<ImplBlock> {
        self.def_id.impl_block(db)
    }
}

impl FnSignature {
    pub(crate) fn fn_signature_query(db: &impl HirDatabase, def_id: DefId) -> Arc<FnSignature> {
        // FIXME: we're using def_id_to_ast here to avoid returning Cancelable... this is a bit hacky
        let node: TreeArc<ast::FnDef> = def_id_to_ast(db, def_id).1;
        let name = node
            .name()
            .map(|n| n.as_name())
            .unwrap_or_else(Name::missing);
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

        let comments = node.doc_comment_text();

        let sig = FnSignature {
            name,
            params,
            ret_type,
            has_self_param,
            documentation: comments,
        };
        Arc::new(sig)
    }
}
