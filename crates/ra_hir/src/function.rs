mod scope;

use std::sync::Arc;

use ra_db::Cancelable;
use ra_syntax::{
    TreePtr,
    ast::{self, AstNode},
};

use crate::{DefId, DefKind, HirDatabase, ty::InferenceResult, Module, Crate, impl_block::ImplBlock, expr::{Body, BodySyntaxMapping}, type_ref::{TypeRef, Mutability}, Name};

pub use self::scope::{FnScopes, ScopesWithSyntaxMapping};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    def_id: DefId,
}

impl Function {
    pub(crate) fn new(def_id: DefId) -> Function {
        Function { def_id }
    }

    pub fn def_id(&self) -> DefId {
        self.def_id
    }

    pub fn syntax(&self, db: &impl HirDatabase) -> TreePtr<ast::FnDef> {
        let def_loc = self.def_id.loc(db);
        assert!(def_loc.kind == DefKind::Function);
        let syntax = db.file_item(def_loc.source_item_id);
        ast::FnDef::cast(&syntax).unwrap().to_owned()
    }

    pub fn body(&self, db: &impl HirDatabase) -> Cancelable<Arc<Body>> {
        db.body_hir(self.def_id)
    }

    pub fn body_syntax_mapping(&self, db: &impl HirDatabase) -> Cancelable<Arc<BodySyntaxMapping>> {
        db.body_syntax_mapping(self.def_id)
    }

    pub fn scopes(&self, db: &impl HirDatabase) -> Cancelable<ScopesWithSyntaxMapping> {
        let scopes = db.fn_scopes(self.def_id)?;
        let syntax_mapping = db.body_syntax_mapping(self.def_id)?;
        Ok(ScopesWithSyntaxMapping {
            scopes,
            syntax_mapping,
        })
    }

    pub fn signature(&self, db: &impl HirDatabase) -> Arc<FnSignature> {
        db.fn_signature(self.def_id)
    }

    pub fn infer(&self, db: &impl HirDatabase) -> Cancelable<Arc<InferenceResult>> {
        db.infer(self.def_id)
    }

    pub fn module(&self, db: &impl HirDatabase) -> Cancelable<Module> {
        self.def_id.module(db)
    }

    pub fn krate(&self, db: &impl HirDatabase) -> Cancelable<Option<Crate>> {
        self.def_id.krate(db)
    }

    /// The containing impl block, if this is a method.
    pub fn impl_block(&self, db: &impl HirDatabase) -> Cancelable<Option<ImplBlock>> {
        self.def_id.impl_block(db)
    }
}

/// The declared signature of a function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnSignature {
    args: Vec<TypeRef>,
    ret_type: TypeRef,
}

impl FnSignature {
    pub fn args(&self) -> &[TypeRef] {
        &self.args
    }

    pub fn ret_type(&self) -> &TypeRef {
        &self.ret_type
    }
}

pub(crate) fn fn_signature(db: &impl HirDatabase, def_id: DefId) -> Arc<FnSignature> {
    let func = Function::new(def_id);
    let node = func.syntax(db);
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
