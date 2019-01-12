//! Many kinds of items or constructs can have generic parameters: functions,
//! structs, impls, traits, etc. This module provides a common HIR for these
//! generic parameters. See also the `Generics` type and the `generics_of` query
//! in rustc.

use std::sync::Arc;

use ra_syntax::ast::{TypeParamList, AstNode, NameOwner};

use crate::{db::HirDatabase, DefId, Name, AsName};

/// Data about a generic parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GenericParam {
    pub(crate) idx: u32,
    pub(crate) name: Name,
}

/// Data about the generic parameters of a function, struct, impl, etc.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Generics {
    pub(crate) params: Vec<GenericParam>,
}

impl Generics {
    pub(crate) fn generics_query(db: &impl HirDatabase, def_id: DefId) -> Arc<Generics> {
        let (_file_id, node) = def_id.source(db);
        let mut generics = Generics::default();
        if let Some(type_param_list) = node.children().find_map(TypeParamList::cast) {
            for (idx, type_param) in type_param_list.type_params().enumerate() {
                let name = type_param
                    .name()
                    .map(AsName::as_name)
                    .unwrap_or_else(Name::missing);
                let param = GenericParam {
                    idx: idx as u32,
                    name,
                };
                generics.params.push(param);
            }
        }
        Arc::new(generics)
    }

    pub(crate) fn find_by_name(&self, name: &Name) -> Option<&GenericParam> {
        self.params.iter().find(|p| &p.name == name)
    }
}
