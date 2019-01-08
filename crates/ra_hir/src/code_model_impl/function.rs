mod scope;

use std::sync::Arc;

use ra_db::Cancelable;
use ra_syntax::{
    TreePtr,
    ast::{self, AstNode},
};

use crate::{
    DefId, DefKind, HirDatabase, Name, Function, FnSignature, Module,
    type_ref::{TypeRef, Mutability},
    expr::Body,
    impl_block::ImplBlock,
};

pub use self::scope::{FnScopes, ScopesWithSyntaxMapping};

impl Function {
    pub(crate) fn new(def_id: DefId) -> Function {
        Function { def_id }
    }

    pub(crate) fn source_impl(&self, db: &impl HirDatabase) -> TreePtr<ast::FnDef> {
        let def_loc = self.def_id.loc(db);
        assert!(def_loc.kind == DefKind::Function);
        let syntax = db.file_item(def_loc.source_item_id);
        ast::FnDef::cast(&syntax).unwrap().to_owned()
    }

    pub(crate) fn body(&self, db: &impl HirDatabase) -> Cancelable<Arc<Body>> {
        db.body_hir(self.def_id)
    }

    pub(crate) fn module(&self, db: &impl HirDatabase) -> Cancelable<Module> {
        self.def_id.module(db)
    }

    /// The containing impl block, if this is a method.
    pub(crate) fn impl_block(&self, db: &impl HirDatabase) -> Cancelable<Option<ImplBlock>> {
        self.def_id.impl_block(db)
    }
}

impl FnSignature {
    pub(crate) fn fn_signature_query(db: &impl HirDatabase, def_id: DefId) -> Arc<FnSignature> {
        let func = Function::new(def_id);
        let node = func.source(db);
        let mut args = Vec::new();
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
                args.push(self_type);
            }
            for param in param_list.params() {
                let type_ref = TypeRef::from_ast_opt(param.type_ref());
                args.push(type_ref);
            }
        }
        let ret_type = if let Some(type_ref) = node.ret_type().and_then(|rt| rt.type_ref()) {
            TypeRef::from_ast(type_ref)
        } else {
            TypeRef::unit()
        };
        let sig = FnSignature { args, ret_type };
        Arc::new(sig)
    }
}
