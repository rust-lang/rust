use std::sync::Arc;

use hir_expand::name::{self, AsName, Name};
use ra_syntax::ast::{self, NameOwner, TypeAscriptionOwner};

use crate::{
    db::DefDatabase2,
    type_ref::{Mutability, TypeRef},
    FunctionId, HasSource, Lookup,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionData {
    pub name: Name,
    pub params: Vec<TypeRef>,
    pub ret_type: TypeRef,
    /// True if the first param is `self`. This is relevant to decide whether this
    /// can be called as a method.
    pub has_self_param: bool,
}

impl FunctionData {
    pub(crate) fn fn_data_query(db: &impl DefDatabase2, func: FunctionId) -> Arc<FunctionData> {
        let src = func.lookup(db).source(db);
        let name = src.value.name().map(|n| n.as_name()).unwrap_or_else(Name::missing);
        let mut params = Vec::new();
        let mut has_self_param = false;
        if let Some(param_list) = src.value.param_list() {
            if let Some(self_param) = param_list.self_param() {
                let self_type = if let Some(type_ref) = self_param.ascribed_type() {
                    TypeRef::from_ast(type_ref)
                } else {
                    let self_type = TypeRef::Path(name::SELF_TYPE.into());
                    match self_param.kind() {
                        ast::SelfParamKind::Owned => self_type,
                        ast::SelfParamKind::Ref => {
                            TypeRef::Reference(Box::new(self_type), Mutability::Shared)
                        }
                        ast::SelfParamKind::MutRef => {
                            TypeRef::Reference(Box::new(self_type), Mutability::Mut)
                        }
                    }
                };
                params.push(self_type);
                has_self_param = true;
            }
            for param in param_list.params() {
                let type_ref = TypeRef::from_ast_opt(param.ascribed_type());
                params.push(type_ref);
            }
        }
        let ret_type = if let Some(type_ref) = src.value.ret_type().and_then(|rt| rt.type_ref()) {
            TypeRef::from_ast(type_ref)
        } else {
            TypeRef::unit()
        };

        let sig = FunctionData { name, params, ret_type, has_self_param };
        Arc::new(sig)
    }
}
