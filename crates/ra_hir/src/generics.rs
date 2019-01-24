//! Many kinds of items or constructs can have generic parameters: functions,
//! structs, impls, traits, etc. This module provides a common HIR for these
//! generic parameters. See also the `Generics` type and the `generics_of` query
//! in rustc.

use std::sync::Arc;

use ra_syntax::ast::{self, AstNode, NameOwner, TypeParamsOwner};

use crate::{db::HirDatabase, DefId, Name, AsName, Function};

/// Data about a generic parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GenericParam {
    pub(crate) idx: u32,
    pub(crate) name: Name,
}

/// Data about the generic parameters of a function, struct, impl, etc.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct GenericParams {
    pub(crate) params: Vec<GenericParam>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum GenericDef {
    Function(Function),
    Def(DefId),
}

impl From<Function> for GenericDef {
    fn from(func: Function) -> GenericDef {
        GenericDef::Function(func)
    }
}

impl From<DefId> for GenericDef {
    fn from(def_id: DefId) -> GenericDef {
        GenericDef::Def(def_id)
    }
}

impl GenericParams {
    pub(crate) fn generic_params_query(
        db: &impl HirDatabase,
        def: GenericDef,
    ) -> Arc<GenericParams> {
        let mut generics = GenericParams::default();
        match def {
            GenericDef::Function(func) => {
                let (_, fn_def) = func.source(db);
                if let Some(type_param_list) = fn_def.type_param_list() {
                    generics.fill(type_param_list)
                }
            }
            GenericDef::Def(def_id) => {
                let (_file_id, node) = def_id.source(db);
                if let Some(type_param_list) = node.children().find_map(ast::TypeParamList::cast) {
                    generics.fill(type_param_list)
                }
            }
        }

        Arc::new(generics)
    }

    fn fill(&mut self, params: &ast::TypeParamList) {
        for (idx, type_param) in params.type_params().enumerate() {
            let name = type_param
                .name()
                .map(AsName::as_name)
                .unwrap_or_else(Name::missing);
            let param = GenericParam {
                idx: idx as u32,
                name,
            };
            self.params.push(param);
        }
    }

    pub(crate) fn find_by_name(&self, name: &Name) -> Option<&GenericParam> {
        self.params.iter().find(|p| &p.name == name)
    }
}
