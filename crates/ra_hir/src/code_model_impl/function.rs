mod scope;

use std::sync::Arc;

use ra_syntax::ast::{self, NameOwner};

use crate::{
    HirDatabase, Name, AsName, Function, FnSignature, Module, HirFileId,
    type_ref::{TypeRef, Mutability},
    expr::Body,
    impl_block::ImplBlock,
    ids::FunctionLoc,
};

pub use self::scope::{FnScopes, ScopesWithSyntaxMapping, ScopeEntryWithSyntax};

impl Function {
    pub(crate) fn from_ast(
        db: &impl HirDatabase,
        module: Module,
        file_id: HirFileId,
        ast: &ast::FnDef,
    ) -> Function {
        let loc: FunctionLoc = FunctionLoc::from_ast(db, module, file_id, ast);
        let id = loc.id(db);
        Function { id }
    }

    pub(crate) fn body(&self, db: &impl HirDatabase) -> Arc<Body> {
        db.body_hir(*self)
    }

    pub(crate) fn module(&self, db: &impl HirDatabase) -> Module {
        self.id.loc(db).module
    }

    /// The containing impl block, if this is a method.
    pub(crate) fn impl_block(&self, db: &impl HirDatabase) -> Option<ImplBlock> {
        let module_impls = db.impls_in_module(self.module(db));
        ImplBlock::containing(module_impls, (*self).into())
    }
}

impl FnSignature {
    pub(crate) fn fn_signature_query(db: &impl HirDatabase, func: Function) -> Arc<FnSignature> {
        let (_, node) = func.source(db);
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

        let sig = FnSignature {
            name,
            params,
            ret_type,
            has_self_param,
        };
        Arc::new(sig)
    }
}
